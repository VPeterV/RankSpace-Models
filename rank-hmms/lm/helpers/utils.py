import time
import os
import logging
from distutils.dir_util import copy_tree
import numpy as np
import random
import torch
from torch.utils.checkpoint import checkpoint as ckp
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.nn.init import xavier_normal_, kaiming_normal_, xavier_uniform_


def logspace(x, eps = 1e-9):
	return (x+ eps).log()

def set_seed(seed):
	"""Sets random seed everywhere."""
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	
def checkpoint(func):
	'''
	To save memory
	'''
	def wrapper(*args, **kwargs):
		return ckp(func, *args, **kwargs)
	return wrapper
	
class Pack(dict):
	def __getattr__(self, name):
		return self[name]

	def add(self, **kwargs):
		for k, v in kwargs.items():
			self[k] = v

	def copy(self):
		pack = Pack()
		for k, v in self.items():
			if type(v) is list:
				pack[k] = list(v)
			else:
				pack[k] = v
		return pack

def get_optimizer(args, model):
	if args.name == 'adam':
		return Adam(params=model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
	elif args.name == 'adamw':
		return AdamW(params=model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
	elif args.optimizer == "sgd":
		optimizer = SGD(
			model.parameters(),
			lr = args.lr,
		)
	else:
		raise NotImplementedError
		
def get_scheduler(args, optimizer):
	if args.name == "reducelronplateau":
		scheduler = ReduceLROnPlateau(
			optimizer,
			factor = 1. / args.decay,
			patience = args.patience,
			verbose = True,
			mode = "max",
		)
	elif args.name == "noam":
		warmup_steps = args.warmup_steps
		def get_lr(step):
			scale = warmup_steps ** 0.5 * min(step ** (-0.5), step * warmup_steps ** (-1.5))
			return args.lr * scale
		scheduler = LambdaLR(
			optimizer,
			get_lr,
			last_epoch=-1,
			verbse = True,
		)
	else:
		raise ValueError("Invalid schedule options")
		
	return scheduler

def get_logger(args, log_name='train',path=None):
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
	handler.setLevel(logging.INFO)
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(formatter)
	logger.addHandler(console)
	logger.propagate = False
	logger.info(args)
	return logger


def create_save_path(args):
	model_name = args.model.model_name
	suffix = "/{}".format(model_name) + time.strftime("%Y-%m-%d-%H_%M_%S",
																			 time.localtime(time.time()))
	from pathlib import Path
	saved_name = Path(args.save_dir).stem + suffix
	args.save_dir = args.save_dir + suffix

	if os.path.exists(args.save_dir):
		print(f'Warning: the folder {args.save_dir} exists.')
	else:
		print('Creating {}'.format(args.save_dir))
		os.makedirs(args.save_dir)
	# save the config file and model file.
	import shutil
	shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
	os.makedirs(args.save_dir + "/lm")
	copy_tree("lm/", args.save_dir + "/lm")
	return  saved_name
	
def get_mask_lengths(text, V):
	mask = text != V.stoi["<pad>"]
	lengths = mask.sum(-1)
	n_tokens = mask.sum()
	return mask, lengths, n_tokens

def log_eye(K, dtype, device):
	x = torch.empty(K, K, dtype = dtype, device = device)
	x.fill_(float("-inf"))
	x.diagonal().fill_(0)
	return x
	
def weight_initializer(model, method = 'normal', special_weights = []):
	for name, param in model.named_parameters():
		# if param.dim() > 1 and 'emb' not in name and 'proj' not in name:
		cons_init = False
		for item in special_weights:
			if item in name:
				cons_init = True
		if param.dim() > 1 and not cons_init:
		# if param.dim() > 1 and 'emb' not in name:
			if method.lower() == 'normal':
				pass
			elif method.lower() == 'xavier_uniform':
				xavier_uniform_(param)
			elif method.lower() == 'xavier_normal':
				xavier_normal_(param)
			elif method.lower() == 'kaiming':
				kaiming_normal_(param)
			else:
				raise AttributeError('Unsupported initialization method')

def get_name(config):
	return "_".join([
		config.version,
		config.data.dataset,
		config.iterator,
		f"db{config.debug}",
		config.model.model_name,
		f"b{config.train.train_bsz}",
		f"m{config.model.num_classes}",
		f"r{config.model.rank}",
		f"dp{config.model.dropout}",
		f"h{config.model.hidden_dim}",
		f"init{config.init}",
		f"cw{config.special_weights}",
		f"c{config.train.clip}",
		config.optimizer.name,
		f"lr{config.optimizer.lr}",
		f"sch{config.scheduler.name}",
		f"tw{config.model.tw}",
	])
	
def get_total_iter(iter):
	cnt = 0
	for _ in iter:
		cnt += 1
		
	return cnt