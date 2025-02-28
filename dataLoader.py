import numpy, os, random, soundfile, torch

def init_loader(args):
	args.trainLoader = torch.utils.data.DataLoader(train_loader(set_type = 'train', **vars(args)), batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	args.valLoader   = torch.utils.data.DataLoader(train_loader(set_type = 'val', **vars(args)),  batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = True)
	args.testLoader  = torch.utils.data.DataLoader(test_loader(**vars(args)), batch_size = 1, shuffle = False, num_workers = 0, drop_last = False)
	return args

def load_audio(path, length):
	audio, _ = soundfile.read(path)
	maxAudio = int(length * 16000)
	if audio.shape[0] < maxAudio:
		shortage  = maxAudio - audio.shape[0]
		audio     = numpy.pad(audio, (0, shortage), 'wrap')
	audio = audio[:maxAudio]    
	return audio

def load_visual(path, length):
	face = numpy.load(path)
	length = int(length * 25)	
	if face.shape[0] < length:
		face = numpy.pad(face, ((0,int(length - face.shape[0])),(0,0)), mode = 'edge')
	face = face[:length,:]
	return face

def audio_overlap(label, infer1, snr1, infer2, snr2, noise, snrn, addition_speaker, addition_noise):
	label_db    = 10 * numpy.log10(numpy.mean(label ** 2)+1e-4)
	infer1_db   = 10 * numpy.log10(numpy.mean(infer1 ** 2)+1e-4)
	infer1      = numpy.sqrt(10 ** ((label_db - infer1_db - snr1) / 10)) * infer1
	audio       = label + infer1

	if addition_speaker == True: # S+S
		infer2_db   = 10 * numpy.log10(numpy.mean(infer2 ** 2)+1e-4)
		infer2      = numpy.sqrt(10 ** ((label_db - infer2_db - snr2) / 10)) * infer2
		audio   = audio + infer2
	if addition_noise == True: # S+S+N
		noise_db    = 10 * numpy.log10(numpy.mean(noise ** 2)+1e-4)
		noise       = numpy.sqrt(10 ** ((label_db - noise_db - snrn) / 10)) * noise   
		audio   = audio + noise
	return audio

class train_loader(object):
	def __init__(self, set_type, data_list, visual_path, audio_path, length, musan_path, **kwargs):
		self.visual_path = visual_path
		self.audio_path = audio_path
		self.length = length
		self.data_list = []
		# If you want to use the musan dataset for augmentation, you can uncomment the following lines and download MUSAN dataset based on [https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py]
		# self.musan_path = musan_path
		# self.augment_files = glob.glob(os.path.join(self.musan_path, 'music/*/*/*.wav')) + glob.glob(os.path.join(self.musan_path, 'noise/*/*/*.wav'))
		lines = open(data_list).read().splitlines()
		dictkeys = []
		for line in lines:
			data = line.split(',')
			if data[0] == set_type:
				dictkeys.append(data[2])
		dictkeys = list(set(dictkeys))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys)}

		label_names = []
		for index, line in enumerate(lines):
			data = line.split(',')
			if data[0] == set_type:
				speaker_label = dictkeys[data[2]]
				label_name = data[2] + '/' + data[3]
				label_names.append(label_name)
				inter_name1 = data[6] + '/' + data[7]
				snr1 = round(float(data[8]), 3)
				length = float(data[-1])
				self.data_list.append([label_name, inter_name1, snr1, length, speaker_label])

	def __getitem__(self, index):         
		label_name, inter_name1, snr1, all_length, speaker_id = self.data_list[index]
		# Load these files
		label   = load_audio(path = os.path.join(self.audio_path, label_name + '.wav'), length = all_length)
		inter_1 = load_audio(path = os.path.join(self.audio_path, inter_name1 + '.wav'), length = all_length)
		face    = load_visual(path = os.path.join(self.visual_path, label_name + '.npy'), length = all_length)    		
		# Generate the mixture
		inter_2, snr2, noise, snrn = None, None, None, None
		audio  = audio_overlap(label, inter_1, snr1, inter_2, snr2, noise, snrn, addition_speaker = False, addition_noise = False) # If you want to simulate more noisy condition, can set the addition_speaker and addition_noise
		noise = audio - label
		# Select the length
		start_face = int(random.random()*((all_length - self.length) * 25))
		# start_face = 0
		start_audio = start_face * 640
		audio = audio[start_audio:start_audio + int(self.length * 16000)]
		label = label[start_audio:start_audio + int(self.length * 16000)]
		noise = noise[start_audio:start_audio + int(self.length * 16000)]
		face = face[start_face:start_face + int(self.length * 25)]
		audio = numpy.divide(audio, numpy.max(numpy.abs(audio)))
		label = numpy.divide(label, numpy.max(numpy.abs(label)))
		noise = numpy.divide(noise, numpy.max(numpy.abs(noise)))
			
		return torch.FloatTensor(audio), \
			   torch.FloatTensor(face), \
			   torch.FloatTensor(label), \
			   torch.FloatTensor(noise), \
			   speaker_id         

	def __len__(self):
		return len(self.data_list)

class test_loader(object):
	def __init__(self, data_list, audio_path, visual_path, musan_path, **kwargs):
		self.audio_path = audio_path
		self.visual_path = visual_path
		# self.musan_path = musan_path
		# self.augment_files = glob.glob(os.path.join(self.musan_path, 'music/*/*/*.wav')) + glob.glob(os.path.join(self.musan_path, 'noise/*/*/*.wav'))
		
		self.data_list = []
		lines = open(data_list).read().splitlines()
		
		for line in lines:
			data = line.split(',')
			data_type = data[0]
			inter_name2, snr2 = None, None
			if data_type == 'test':			
				label_name = data[2] + '/' + data[3]
				inter_name1 = data[6] + '/' + data[7]
				snr1 = round(float(data[8]), 3)
				length = float(data[-1])
				self.data_list.append([label_name, inter_name1, snr1, inter_name2, snr2, length])

	def __getitem__(self, index):        
		label_name, inter_name1, snr1, inter_name2, snr2, all_length = self.data_list[index]
		# Load these files
		label   = load_audio(path = os.path.join(self.audio_path, label_name + '.wav'), length = all_length)
		inter_1 = load_audio(path = os.path.join(self.audio_path, inter_name1 + '.wav'), length = all_length)
		face    = load_visual(path = os.path.join(self.visual_path, label_name + '.npy'), length = all_length)    
		snrn    = random.uniform(5,15)
		# Generate the mixture
		inter_2, noise = None, None
		audio  = audio_overlap(label, inter_1, snr1, inter_2, snr2, noise, snrn, addition_speaker = False, addition_noise = False)

		noise = audio - label
		audio = numpy.divide(audio, numpy.max(numpy.abs(audio)))
		label = numpy.divide(label, numpy.max(numpy.abs(label)))
		noise = numpy.divide(noise, numpy.max(numpy.abs(noise)))
			
		return torch.FloatTensor(audio), \
			   torch.FloatTensor(face), \
			   torch.FloatTensor(label), \
			   torch.FloatTensor(noise), \
			   torch.FloatTensor(noise)

	def __len__(self):
		return len(self.data_list)

