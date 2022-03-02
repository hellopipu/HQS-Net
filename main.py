# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import argparse
from Solver import Solver


def main(args):
    print(args)
    solver = Solver(args)
    if args.mode == 'test':
        solver.test()
    elif args.mode == 'train':
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ############################### experiment settings ##########################
    parser.add_argument('--mode', default='train', choices=['train', 'test'],
                        help='mode for the program')
    parser.add_argument('--model', default='hqs-net',
                        choices=['dc-cnn', 'lpd-net', 'hqs-net', 'hqs-net-unet', 'ista-net-plus'],
                        help='models to reconstruct')
    parser.add_argument('--acc', type=int, default=5,
                        help='Acceleration factor for k-space sampling')
    ############################### dataset setting ###############################

    parser.add_argument('--train_path', default="data/fs_train.npy",
                        help='train_path')
    parser.add_argument('--val_path', default="data/fs_val.npy",
                        help='val_path')
    parser.add_argument('--test_path', default="data/fs_test.npy",
                        help='test_path')

    ############################### model training settings ########################
    parser.add_argument('--num_epoch', type=int, default=300,
                        help='num of training epoch')
    parser.add_argument('--val_on_epochs', type=int, default=1,
                        help='validate for each n epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size, 1,4,8,16, ...')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for training')
    parser.add_argument('--resume', type=int, default=0, choices=[0, 1],
                        help='resume training')

    args = parser.parse_args()

    main(args)
