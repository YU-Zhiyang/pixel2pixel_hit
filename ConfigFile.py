# Dataset Related
import os
import logging
import torch
from datetime import datetime
from utils.FileOption import make_dir_tree
from tensorboardX import SummaryWriter


class Config():
    def __init__(self):
        # state
        self.isTrain = True
        self.continue_train = False
        self.random_seed = 2019

        # train data and log
        self.dataroot = './dataset'
        self.output_dir = './output'
        self.checkpoints_dir = None
        self.outputpath_init()
        self.events_writer = SummaryWriter(self.events_dir)

        # hardware
        self.gpu_ids = [1]
        self.set_gpu_ids()

        # network set
        self.norm = 'instance'
        self.netG = 'unet_256'
        self.netD = 'basic'
        self.gan_mode = 'lsgan'
        self.ngf = 64
        self.ndf = 64
        self.n_layers_D = 3
        self.init_gain = 0.02
        self.init_type = 'normal'
        self.input_nc = 3
        self.output_nc = 3
        self.pool_size = 50
        self.no_dropout = False

        # train set
        self.epoch = 10
        self.batch_size = 1

        self.snapshot = 5000
        # learning rate
        self.lr = 0.0002
        self.lr_decay_iters = 50
        self.lr_policy = 'step'
        self.niter = 100
        self.niter_decay = 100

        # optimizer
        self.beta1 = 0.5
        self.lambda_L1 = 100.0
        self.lambda_D = 1
        self.lambda_G = 1

    def set_gpu_ids(self):
        str_ids = self.gpu_ids
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

    def outputpath_init(self):
        if self.checkpoints_dir is None:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.checkpoints_dir = os.path.join(self.output_dir, '{}'.format(current_time))
            self.events_dir = os.path.join(self.checkpoints_dir, 'events')
            self.model_dir = os.path.join(self.checkpoints_dir, 'model')
            self.image_dir = os.path.join(self.checkpoints_dir, 'image')
            make_dir_tree(self.checkpoints_dir)
            make_dir_tree(self.events_dir)
            make_dir_tree(self.model_dir)
            make_dir_tree(self.image_dir)
        else:
            self.checkpoints_dir = os.path.join(self.output_dir, self.checkpoints_dir)
            self.events_dir = os.path.join(self.checkpoints_dir, 'events')
            self.model_dir = os.path.join(self.checkpoints_dir, 'model')
            self.image_dir = os.path.join(self.checkpoints_dir, 'image')

        self.logger = logging.getLogger(__name__)
        logfile = os.path.join(self.checkpoints_dir, 'log.txt')
        fh = logging.FileHandler(logfile, mode='a')
        # fh = logging.StreamHandler(sys.stdout)
        # fh = logging.NullHandler()
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def record(self):
        logger_ = logging.getLogger(__name__+'sub')
        logging.basicConfig(level=logging.INFO)
        path = os.path.join(self.checkpoints_dir, 'config.txt')
        fh = logging.FileHandler(path, mode='w')
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger_.addHandler(fh)

        args = vars(self)
        logger_.info('-------------Config-------------------')
        for k, v in sorted(args.items()):
            logger_.info('{} = {}'.format(k, v))
        logger_.info('--------------End---------------------')
        # if save:
        #     with open(path, 'wt') as conf_file:
        #         conf_file.write('-------------Config-------------------\n')
        #         for k, v in sorted(args.items()):
        #             conf_file.write('{} = {} \n'.format(k, v))
        #         conf_file.write('--------------End---------------------')


if __name__ == '__main__':
    cfg = Config()
    pass
