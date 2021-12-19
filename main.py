# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import argparse
from Solver import Solver


def main(args):
    print(args)
    solver = Solver(args)
    if args.mode == 'visualize':
        solver.visualize()
    elif args.mode == 'test':
        solver.test()
    elif args.mode == 'train':
        solver.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ############################### experiment settings ###############################
    parser.add_argument('--mode', default='train', choices=['train', 'test','visualize'],
                        help='mode for the program')
    parser.add_argument('--expname', default='',
                        help='expname for the program, used for develop  model')
    parser.add_argument('--dataset', default='knee_100', choices=['knee_100', 'ocmr','calgary'],
                        help='mode for the program')
    parser.add_argument('--model', default='refinegan', choices=['refinegan', 'd5c5','refinegan_d5c5','pdnet','refinegan_lpd', 'refine_G','refine_lpd','hybrid','refinegan_lpd_im','miccan','dense_pdnet'],
                        help='model for the program')
    parser.add_argument('--mask_type', default='cartes', choices=['radial', 'cartes', 'gauss', 'spiral'],
                        help='mask type')
    parser.add_argument('--loss', default='mse', choices=['mse','compound','ssim','l1','compound2'],
                        help='model for the program')
    parser.add_argument('--data_aug', type=int, default=1, choices=[0,1],
                        help='when training: 0, no aug; 1, aug')
    ############################### dataset setting ###############################
    # for ocmr dataset only
    parser.add_argument('--acc', type=int, default=8,
                        help='Acceleration factor for k-space sampling, for ocmr dataset only')
    ## for ixi only;
    parser.add_argument('--sampling_rate', type=int, default=10,
                        help='for knee_100, sampling rate for mask, only 10, 20,30,40, ...')

    ############################### path settings ###############################
    parser.add_argument('--train_path', default='data/knees/db_train/',
                        help='train_path')
    parser.add_argument('--val_path', default='data/knees/db_valid/',
                        help='val_path')
    parser.add_argument('--test_path', default='data/knees/db_valid/',
                        help='test_path')

    ############################### model training settings ###############################
    parser.add_argument('--num_epoch', type=int, default=500,
                        help='num of training epoch')
    parser.add_argument('--val_on_epochs', type=int, default=1,
                        help='val for each n epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size, 4,8,16, ...')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for training')
    parser.add_argument('--end_factor', type=float, default=6e-2,
                        help='learning rate for training')
    parser.add_argument('--norm_type', type=int, default=0, choices=[0,1],
                        help='norm_type == 0, range : [0,1]; norm_type == 1, refinegan norm, range : [-1,1]')
    parser.add_argument('--beta', type=int, default=0, choices=[0,1],
                        help='beta == 0, default beta; beta == 1, refinegan beta, (0.5,0.999)')

    args = parser.parse_args()

    main(args)

