# -*- coding: utf-8 -*-
import nni
import wandb
import argparse
import os
import shutil
import torch
import traceback
from pathlib import Path
from easydict import EasyDict as edict
import yaml
from cmds.train import Train
from lm.helpers.utils import get_name, set_seed

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='HMMLM'
	)
	parser.add_argument('--conf', '-c', default='')
	parser.add_argument('--device', '-d', default='0')
	parser.add_argument('--version', default="projhmmlm")

	args2 = parser.parse_args()
	yaml_cfg = yaml.load(open(args2.conf, 'r'), Loader = yaml.FullLoader)
	args = edict(yaml_cfg)
	args.update(args2.__dict__)
 
	print(f"Set the device with ID {args.device} visible")
	os.environ['CUDA_VISIBLE_DEVICES'] = args.device
	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	set_seed(args.seed)
	
	config_name = get_name(args)
	args.save_dir = args.save_dir + "/{}".format(config_name)
	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	
	if args.wandb:
		wandb.init(project="rankhmmlm", name=config_name, config=args, dir=args.wandb_log, mode=args.wandb_mode)

	try:
		command = Train()
		command(args)
	except KeyboardInterrupt:
		command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
		if command == 0:
			shutil.rmtree(args.save_dir)
			print("You have successfully delete the created log directory.")
		else:
			print("log directory have been saved.")
	except Exception:
		traceback.print_exc()
		shutil.rmtree(args.save_dir)
		print("log directory have been deleted.")

