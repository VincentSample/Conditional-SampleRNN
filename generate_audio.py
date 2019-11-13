#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 18:40:09 2019

@author: vincent
"""
import os
from model import SampleRNN, Generator
from librosa.output import write_wav
import torch
import argparse
import sys
from trainer.plugins import GeneratorPlugin
#model = SampleRNN([16, 4], 2, 1024, True, 256, True)


def main(exp, frame_sizes, generate_from,**params):
    params = dict(
        default_params,
        exp=exp, frame_sizes=frame_sizes,generate_from=generate_from, 
        **params
    )
    model = SampleRNN(
        frame_sizes=params['frame_sizes'],
        n_rnn=params['n_rnn'],
        dim=params['dim'],
        learn_h0=params['learn_h0'],
        q_levels=params['q_levels'],
        nb_classes=params['nb_classes'],
        weight_norm=params['weight_norm'],
            
    )
#    model = SampleRNN([16, 4], 2, 1024, True, 256, True)
    print('Loading saved model' + params['generate_from'] ) 
    checkpoint = torch.load(params['generate_from'])
    temporary_dict={}
    for k, v in checkpoint.items():
        temporary_dict[k[6:]] = v
    checkpoint = temporary_dict
    model.load_state_dict(checkpoint)
    if not os.path.exists(params['generate_to']):
        os.mkdir(params['generate_to'])
    print(params['cond'])
    generator = GeneratorPlugin(params['generate_to'], params['n_samples'], params['sample_length'], params['sample_rate'], params['nb_classes'], params['cond'] )
    generator.register_generate(model.cuda(), params['cuda'])
    generator.epoch(exp) 

default_params = {
    # model parameters
    'n_rnn': 2,
    'dim': 1024,
    'learn_h0': True,
    'q_levels': 256,
    'seq_len': 1024,
    'weight_norm': True,
    'batch_size': 128,
    'val_frac': 0.1,
    'test_frac': 0.1,

    # training parameters
    'keep_old_checkpoints': False,
    'results_path': 'results',
    'epoch_limit': 1000,
    'resume': True,
    'sample_rate': 16000,
    'n_samples': 3,
    'sample_length': 100,
    'loss_smoothing': 0.99,
    'cuda': True,
    'comet_key': None,
    'generate_to' : 'results',
    'cond' : False
}

tag_params = [
    'exp', 'frame_sizes', 'n_rnn', 'dim', 'learn_h0', 'q_levels', 'seq_len',
    'batch_size', 'val_frac', 'test_frac', "generate_from"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()

    parser.add_argument('--exp', required=True, help='generationname')
    parser.add_argument(
        '--frame_sizes', nargs='+', type=int, required=True,
        help='frame sizes in terms of the number of lower tier frames, \
              starting from the lowest RNN tier'
    )
    parser.add_argument(
        '--cond', nargs='+', type=int, required=False,
        help='conditioning vector \
                        to generate with, format : 1 0 0 0'
    )
    
    parser.add_argument(
        '--n_rnn', type=int, help='number of RNN layers in each tier'
    )
    parser.add_argument(
        '--dim', type=int, help='number of neurons in every RNN and MLP layer'
    )
    parser.add_argument(
        '--learn_h0', type=parse_bool,
        help='whether to learn the initial states of RNNs'
    )
    parser.add_argument(
        '--q_levels', type=int,
        help='number of bins in quantization of audio samples'
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='how many samples to include in each truncated BPTT pass'
    )
    parser.add_argument(
        '--generate_from', required=True, help='model to generate from'
    )
    parser.add_argument(
        '--generate_to', help='dir to save results'
    )
    parser.add_argument(
        '--sample_rate', type=int,
        help='sample rate of the training data and generated sound'
    )
    parser.add_argument(
        '--n_samples', type=int,
        help='number of samples to generate in each epoch'
    )
    parser.add_argument(
        '--nb_classes', type=int,
        help='number of classes (labels) of training data'
    )
    parser.add_argument(
        '--sample_length', type=int,
        help='length of each generated sample (in samples)'
    )

    parser.add_argument(
        '--cuda', type=parse_bool,
        help='whether to use CUDA'
    )

    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
    
