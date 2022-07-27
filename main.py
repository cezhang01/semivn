import argparse
import numpy as np
import tensorflow as tf
from data_preparation import Data
from model import Model


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-ne', '--num_epoch', type=int, default=300)
    parser.add_argument('-dn', '--dataset_name', type=str, default='coronavirus')
    parser.add_argument('-ra', '--labeling_ratio', type=float, default=0.8)
    parser.add_argument('-nn', '--num_neg', type=float, default=5)
    parser.add_argument('-nt', '--num_topics', type=int, default=30)
    parser.add_argument('-vd', '--visualization_dimensions', type=int, default=2)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=128)
    parser.add_argument('-l', '--regularizer', type=float, default=1)
    parser.add_argument('-ii', '--call_interactive_interface', type=int, default=0)
    parser.add_argument('-rs', '--random_seed', type=int, default=519)

    return parser.parse_args()


def main(args):

    if args.random_seed:
        tf.set_random_seed(args.random_seed)
        np.random.seed(args.random_seed)
    print('Preparing data...')
    data = Data(args)
    print('Initializing model...')
    model = Model(args, data)
    print('Start training...')
    model.train()


if __name__ == '__main__':
    main(parse_arguments())