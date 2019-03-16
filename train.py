from ConfigFile import Config
from models.Pixel2Pixel_model import Pix2PixModel
import torch
import random

if __name__ == '__main__':
    opt = Config()   # get training options
    opt.record()
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    model = Pix2PixModel(opt)
    model.training()