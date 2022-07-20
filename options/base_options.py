import argparse
import os
from util import util
import torch
import models
import data
from datetime import datetime
import sys
import time
import random
import numpy as np
import warnings


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='ckpts', help='models are saved here')
        parser.add_argument('--lucky_seed', type=int, default=0, help='seed for random initialize, 0 to disable, -1 to use current time.')
        # model parameters
        parser.add_argument('--model', type=str, default='iern', help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--emo_num', type=int, default=6, help='# of facial expression types.')
        parser.add_argument('--cnfnd_num', type=int, default=3, help='# of confounders')
        parser.add_argument('--backbone_archi', type=str, default='resnet50', help='specify backbone architecture')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_backbone_pretrained', action='store_true', help='dont load vgg imagenet pretrained weights.')
        # dataset parameters
        parser.add_argument('--dataroot', required=True, help='path to training dataset')
        parser.add_argument('--dataset_name', type=str, default='mixbasic', help='choose which dataset to use. [mixbasic | toyckp | webemo | emotion6 | sentiment | alpha]')
        parser.add_argument('--test_dataroot', type=str, default='null', help='path to testing dataset')
        parser.add_argument('--test_dataset_name', type=str, default='mixbasic', help='choose which dataset to use for testing. [mixbasic | toyckp | webemo | emotion6 | sentiment]')
        parser.add_argument('--train_conf_name', type=str, default='train_ids_0.csv', help='training split image names.')
        parser.add_argument('--test_conf_name', type=str, default='test_ids_0.csv', help='testing split image names.')
        parser.add_argument('--emo_name', type=str, default='emotion_labels.pkl', help='label dictionary file path.')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--test_batch_size', type=int, default=16, help='validate set input batch size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=224, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='common', help='preprocess methods [common | none | light]')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # update default for testing dataset
        # parser.set_defaults(test_dataroot=parser.get_default('dataroot'),
        #                     test_dataset_name=parser.get_default('dataset_name'))

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_name
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # update default for testing dataset
        parser.set_defaults(test_dataroot=opt.dataroot,
                            test_dataset_name=opt.dataset_name,
                            test_batch_size=opt.batch_size)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        expr_dir = opt.checkpoints_dir if opt.isTrain else opt.results_dir
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'a+') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # update checkpoint path
        if opt.isTrain and (not opt.continue_train):
            suffix = datetime.now().strftime("%m%d_%H%M%S")
            fold_id = "fold_%s" % os.path.splitext(opt.train_conf_name)[0].split('_')[-1]
            opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, fold_id, opt.model, suffix)
            util.mkdirs(opt.checkpoints_dir)
        # update result path in test
        if not opt.isTrain:
            suffix = '/'.join(opt.checkpoints_dir.strip('/').split('/')[1:])
            opt.results_dir = os.path.join(opt.results_dir, suffix)
            util.mkdirs(opt.results_dir)

        # write command to file and print option
        expr_dir = opt.checkpoints_dir if opt.isTrain else opt.results_dir
        with open(os.path.join(expr_dir, "run_script.sh"), 'a+') as f:
            f.write("python %s\n" % ' '.join(sys.argv))
        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # set seed, note that although fix the seed can make the result reproductive, using this way may slow down the training...
        if opt.lucky_seed != 0:
            if opt.lucky_seed == -1:
                opt.lucky_seed = int(time.time())
            random.seed(a=opt.lucky_seed)
            np.random.seed(seed=opt.lucky_seed)
            # ref: https://discuss.pytorch.org/t/random-seed-initialization/7854/19
            # num_workers = 0 and torch.backends.cudnn.enabled = False
            opt.num_threads = 0
            if len(opt.gpu_ids) > 0:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False
            torch.manual_seed(opt.lucky_seed)

        self.opt = opt
        return self.opt
