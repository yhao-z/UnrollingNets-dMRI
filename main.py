import argparse
import os
import sys
import tensorflow as tf
from Solver import Solver
from loguru import logger
import random
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

random_seed = 3407 
random.seed(random_seed )  # set random seed for python
np.random.seed(random_seed )  # set random seed for numpy
tf.random.set_seed(random_seed )  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1' # set random seed for tensorflow-gpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU No.')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='ocmr', help='ocmr, cmrxrecon')
    parser.add_argument('--multicoil', type=int, default=1, help='1 true, 0 false')
    parser.add_argument('--masktype', type=str, default='vista', help='radial, vds, vista, fastmri_equispaced')
    parser.add_argument('--acc', type=float, default=8, help='acceleration factor')
    # training parameters
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch, begin with 1')
    parser.add_argument('--end_epoch', type=int, default=50, help='end epoch or number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    # network define
    parser.add_argument('--ModelName', type=str, default='T2LR_Net', help='DUS_Net, DUS_Net_s')
    parser.add_argument('--weight', type=str, default=None, help='*/weight-best')
    # train, test mode choosing and parameters
    parser.add_argument('--mode', type=str, default='test', help='train, test, test_spec')
    parser.add_argument('--autoload', type=int, default=1, help='1:true, 0 false | if autoload pretraining weight')
    parser.add_argument('--spec', type=int, default=2, help='test spec num, which data is testing and saving')
    # debug or not
    parser.add_argument('--debug', type=int, default=1, help='1:debug, 0 running')
    
    args = parser.parse_args()
    
    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)
    
    if args.dataset == 'ocmr':
        args.datadir = '/workspace/DATA/OCMR/tfrecord/sigcoil/' if args.multicoil == 0 else '/workspace/DATA/OCMR/tfrecord/multicoil/'
    elif args.dataset == 'cmrxrecon':
        args.datadir = '/workspace/DATA/CMRxRecon/MultiCoil/Cine/tfrecord/'
    
    solver = Solver(args)
    logger.info(args)
    logger.critical('some words')
    
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test(autoload=args.autoload)
    elif args.mode == 'test_spec':
        solver.test_spec(args.spec, autoload=args.autoload)
    
