# Some expenience for this task

Hi! I want to share some observation for audio-visual target speaker extraction from my experience. Hope can assist your research

## Why this task can work?

Lip reading is linked with speech phoneme: In the noisy environment, we can listen to somebody better if we can see his/her mouth, because lip movements can assist us to extract the speech in this process. 

## What is each sample in this task?

```
    Let's say we have video1 (speech1+visual1) and video2 (speech2+visual2):
    For training/evaluation, the input data is:
    Audio part: speech1 + speech2 -> Mixed speech;
    Visual part: visual1
    Output: estimated speech1 vs ground truth speech1
```

## Issue about synchronicity for the dataset

- Note that in the LRS2, all videos are strickly synchronized. In the VoxCeleb2, audio and visual information has around 1 frame (40ms) out-of-sync.
- So model trained in VoxCeleb2 can fit well for slightly out-of-sync, but the model trained in LRS2 cannot used for these slight out-of-sync videos. 

## About performance

For the results of other tables in our paper, I can provide the basic description of how to conduct them:
- If you want to train for the Table VIII, the non-speech noisy signal requires the MUSAN dataset, I have commented and keep the related code in dataLoader.py
- For other experiments in my paper, I cannot summary all of them while they are not difficult to achieve.
- Meanwhile, to achieve robust results, you can try to train with unlimited samples from VoxCeleb2, arbitrary start_second of audio, arbitrary mixture generation selection.
- During training, fixed mixture with the random segments are selected, during testing, the fixed mixture with the entire segments are selected. 

## Training speed

My device is one RTX-4090, the training time has been printed in the training log files for different model achitectures.

## Relationship between Si-SNR and Si-SNRi

- Easy extraction condition (enhancement, speech + background noise -> speech), 
    - high Si-SNR, low Si-SNRi
- Hard extraction condition (speech + interference + interference + backgound noise -> speech), 
    - low Si-SNR, high Si-SNRi

## For LRS3, TCD and Grid dataset

- We conducted the experiments on VoxCeleb2 and LRS2 (for training and evaluation), LRS3, TCD and Grid (for evaluation only). This project only provides VoxCeleb2 since it is popular.
- The idea about how to conducting experiments for them: Get clean one-person talking video, mix up the audio to generate the mixture of speech.

## Multi-card training

This code can used for multi-card training, it is not difficult and you can easily achieve that. Unfortunally, now I do not have multiply cards for debugging... So I can share you about how I once modified my code. (It can work and I once verified.)

In run.sh: 

```torchrun --nproc_per_node=4 --master_port=29500 main.py```

In main.py, add that in the beginning:

```
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0, 1, 2, 3"
```

In trainer.py, add that in the beginning:

```
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```

Then modifiy the init function in trainer to config DDP

```
class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		dist.init_process_group(backend='nccl', init_method='env://') # For multi-card training
		self.local_rank = int(os.getenv("LOCAL_RANK", 0))
		torch.cuda.set_device(self.local_rank)
		self.model           = LTSE().cuda() # Define the model
		self.model           = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True) # For multi-card training
		self.optim           = torch.optim.AdamW(self.model.parameters(), lr = args.lr) # optimizer
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay) # scheduler
		if dist.get_rank() == 0: # Only print for one time
			print("Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1e6)) # Print the model size
```

In the train_network, add ```if dist.get_rank() == 0:``` before printing.

In dataLoader.py, modify the init_loader with:

```
from torch.utils.data import DataLoader, DistributedSampler

def init_loader(args):
	args.trainLoader = DataLoader(
		train_loader(set_type = 'train', **vars(args)),
		batch_size=args.batch_size,
		sampler=DistributedSampler(train_loader(set_type = 'train', **vars(args))),
		num_workers=args.n_cpu
	)
	args.valLoader = DataLoader(
		val_loader(set_type = 'val', **vars(args)),
		batch_size=args.batch_size,
		sampler=DistributedSampler(train_loader(set_type = 'val', **vars(args))),
		num_workers=args.n_cpu
	)
	args.testLoader = DataLoader(
		test_loader(**vars(args)),
		batch_size=args.batch_size,
		sampler=DistributedSampler(test_loader(**vars(args))),
		num_workers=args.n_cpu
	)
	return args
```

## MULTI-MODAL INTERACTION
I did not include the code about the section ```EXPLORATION OF MULTI-MODAL INTERACTION```