import argparse

import torch

from src.models import PseudoNetBin
from src.trainer import Trainer


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--epochs", type=int, default=100)
arg_parser.add_argument("--batch_size", type=int, default=64)
arg_parser.add_argument("--num_workers", type=int, default=6)
arg_parser.add_argument("--lr", type=float, default=0.0001)
arg_parser.add_argument("--model_weight", type=str)

args = arg_parser.parse_args()

model = PseudoNetBin(args)
if args.model_weight is not None:
    model.load_state_dict(torch.load(args.model_weight))

trainer = Trainer(model, args)
trainer.train_unsup()
