import torch, sys, os, time
import torch.nn as nn
from tools import *
from loss  import *
from model.dprnn import dprnn
from model.muse import muse
from model.avsep import avsep
from model.seanet import seanet
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict 

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.init_model != "":
		print("Model %s loaded from pretrain!"%args.init_model)
		s.load_parameters(args.init_model)
	elif len(args.modelfiles) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles[-1])
	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		if args.backbone == 'seanet':
			self.model       = seanet(256, 40, 64, 128, 100, 6).cuda()
		elif args.backbone == 'avsep':
			self.model       = avsep().cuda()
		elif args.backbone == 'muse':
			self.model       = muse(M = 800).cuda() # Based on your dataset
		elif args.backbone == 'dprnn':
			self.model       = dprnn().cuda()
		self.loss_se     = loss_speech().cuda()
		self.optim       = torch.optim.AdamW(self.parameters(), lr = args.lr)
		self.scheduler   = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.val_step, gamma = args.lr_decay)
		print("Model para number = %.2f"%(sum(param.numel() for param in self.parameters()) / 1e6))
		
	def train_network(self, args):
		B, time_start, nloss, nloss_muse = args.batch_size, time.time(), 0, 0
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		lr = self.optim.param_groups[0]['lr']	
		self.speaker_loss = nn.CrossEntropyLoss()
		for num, (audio, face, speech, noise, muse_label) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			self.speaker_loss = nn.CrossEntropyLoss()
			with autocast():				
				muse_label            = torch.LongTensor(muse_label).cuda()							
				audio, face, speech, noise = audio.cuda(), face.cuda(), speech.cuda(), noise.cuda()
				if args.backbone == 'seanet':
					out_s, out_n = self.model(audio, face, M = B)			
					loss_s_main = self.loss_se.forward(out_s[-B:,:], speech)
					loss_n_main = self.loss_se.forward(out_n[-B:,:], noise)	
					loss_n_rest = self.loss_se.forward(out_n[:-B,:], noise.repeat(5, 1))
					loss_s_rest = self.loss_se.forward(out_s[:-B,:], speech.repeat(5, 1))
					loss = loss_s_main + (loss_n_main + loss_n_rest + loss_s_rest) * args.alpha
				elif args.backbone == 'dprnn':
					out_s = self.model(audio, face, M = B)				
					loss_s_main = self.loss_se.forward(out_s[-B:,:], speech)
					loss_s_rest = self.loss_se.forward(out_s[:-B,:], speech.repeat(5, 1))
					loss = loss_s_main + (loss_s_rest) * args.alpha	
				elif args.backbone == 'muse':
					out_e, out_s = self.model(audio, face)
					loss_s_main = self.loss_se.forward(out_s, speech)
					loss_muse = 0
					for i in range(4):
						loss_muse += self.speaker_loss(out_e[i], muse_label)
					loss = loss_s_main + loss_muse * 0.1
				elif args.backbone == 'avsep':
					out_s = self.model(audio, face)			
					loss = self.loss_se.forward(out_s, speech)

			scaler.scale(loss).backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
			scaler.step(self.optim)
			scaler.update()

			nloss += loss.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.3f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, nloss/B/num))
			sys.stderr.flush()
		sys.stdout.write("\n")

		args.score_file.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.3f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, nloss/B/num))
		args.score_file.flush()
		return


	def eval_network(self, eval_type, args):
		Loader = args.valLoader   if eval_type == 'Val' else args.testLoader
		B      = args.batch_size  if eval_type == 'Val' else 1
		self.eval()
		time_start = time.time()
		sisdr_speech, sdr_speech, sisdri_speech, sdri_speech = 0, 0, 0, 0
		for num, (audio, face, speech, noise, _) in enumerate(Loader, start = 1):
			self.zero_grad()
			with torch.no_grad():
				audio, face, speech = audio.cuda(), face.cuda(), speech.cuda()
				if args.backbone == 'seanet':
					out_speech, _ = self.model(audio, face, B)
					out = out_speech[-B:,:]
				elif args.backbone == 'dprnn':
					out_speech = self.model(audio, face, B)
					out = out_speech[-B:,:]
				elif args.backbone == 'muse':
					_, out_speech = self.model(audio, face)
					out = out_speech
				elif args.backbone == 'avsep':
					out_speech = self.model(audio, face)
					out = out_speech

				res_speech = self.loss_se.forward_eval_light(out, speech)
				res_orig   = self.loss_se.forward_eval_light(audio, speech)
				
			sisdri_speech += (res_speech['sisdr'] - res_orig['sisdr']).detach().cpu().numpy()
			sdri_speech += (res_speech['sdr'] - res_orig['sdr']).detach().cpu().numpy()
			sisdr_speech += (res_speech['sisdr']).detach().cpu().numpy()
			sdr_speech += (res_speech['sdr']).detach().cpu().numpy()

			time_used = time.time() - time_start
			sys.stderr.write("%s: [%2d] %.2f%% (%.1f mins), SISDR: %.3f, SDR: %.3f, SISDRi: %.3f, SDRi: %.3f\r"%\
			(eval_type, args.epoch, 100 * (num / Loader.__len__()), time_used * Loader.__len__() / num / 60, \
			sisdr_speech/num, sdr_speech/num, sisdri_speech/num, sdri_speech/num))
			sys.stderr.flush()
		sys.stdout.write("\n")

		args.score_file.write("%s: [%2d] %.2f%% (%.1f mins), SISDR: %.3f, SDR: %.3f, SISDRi: %.3f, SDRi: %.3f\r"%\
			(eval_type, args.epoch, 100 * (num / Loader.__len__()), time_used * Loader.__len__() / num / 60, \
			sisdr_speech/num, sdr_speech/num, sisdri_speech/num, sdri_speech/num))
		args.score_file.flush()
		return

	def save_parameters(self, path):
		model = OrderedDict(list(self.state_dict().items()))
		torch.save(model, path)

	def load_parameters(self, path):
		selfState = self.state_dict()
		loadedState = torch.load(path)	
		for name, param in loadedState.items():
			origName = name
			if name not in selfState:
				name = 'model.' + name
				if name not in selfState:
					print("%s is not in the model."%origName)
					continue
			if selfState[name].size() != loadedState[origName].size():
				sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
				continue
			selfState[name].copy_(param)