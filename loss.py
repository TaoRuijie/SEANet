import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio import signal_distortion_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

class loss_speech(nn.Module):
    def __init__(self):
        super(loss_speech, self).__init__()

    def forward(self, out_speech, labels_speech):
        loss = cal_SISNR(labels_speech, out_speech)
        loss = -torch.sum(loss)
        return loss

    def forward_eval_light(self, out_speech, labels_speech):    
        sdr  = torch.mean(signal_distortion_ratio(out_speech, labels_speech))
        sisdr = torch.mean(scale_invariant_signal_distortion_ratio(out_speech, labels_speech))       
        res = {'sisdr': sisdr, 'sdr': sdr}
        return res

    def forward_eval_full(self, out_speech, labels_speech): # Can evaluate PESQ and STOI
        sisdr = torch.mean(scale_invariant_signal_distortion_ratio(out_speech, labels_speech))
        sdr  = torch.mean(signal_distortion_ratio(out_speech, labels_speech))
        pesq = torch.mean(perceptual_evaluation_speech_quality(out_speech, labels_speech, 16000, 'wb'))
        stoi = torch.mean(short_time_objective_intelligibility(out_speech, labels_speech, 16000).float().cuda())
        res = {'sisdr': sisdr, 'sdr': sdr, 'pesq': pesq, 'stoi': stoi}
        return res
