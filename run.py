import argparse
import torch
import yaml
import os
from utils.general import join
from src.VAE import VAE


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, type=str)
parser.add_argument("--train", action='store_true')
parser.add_argument("--record", action='store_true')


if __name__== "__main__":
	args = parser.parse_args()

	if args.config is not None:
		yaml.add_constructor("!join", join)
		config_file = open("config/{}.yml".format(args.config))
		config = yaml.load(config_file, Loader=yaml.FullLoader)

	if not os.path.exists(config["output"]["model_output"]):
		os.makedirs(config["output"]["model_output"])

	model = VAE()

	if args.train:
		model.train()
	
	else:
		pass
