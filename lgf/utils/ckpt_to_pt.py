#ckpt_to_pt.py

import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-C', type=str, default='last.ckpt')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    model = ckpt['state_dict']
    model = {k.replace('model.', ''): v for k, v in model.items()}
    torch.save(model, args.ckpt.replace('ckpt', 'pt'))
