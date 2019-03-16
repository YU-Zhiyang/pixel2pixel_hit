import torch
from dataloader.ImageReader import create_dataset

from models import losses
from models import Methods
import os
from ConfigFile import Config
import torchvision.utils as vutils
import numpy as np
import cv2
import torch.nn.functional as F


class Pix2PixModel(object):

    def __init__(self, opt: Config):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.writer = opt.events_writer
        self.logger = opt.logger
        self.epoch_past = 0
        self.total_iter = 0
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids \
            else torch.device('cpu')  # get device name: CPU or GPU

        self.model_names = []
        self.optimizers = []
        self.metric = None  # used for learning rate policy 'plateau'

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = Methods.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.dataset_train, self.dataset_val = create_dataset(opt)
            self.netD = Methods.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.criterionD = losses.DisLoss(opt.gan_mode, weight=opt.lambda_D)
            self.criterionG = losses.GanLoss(opt.gan_mode, weight=opt.lambda_G)
            self.criterionL1 = losses.L1Loss(weight=opt.lambda_L1)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.schedulers = [Methods.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        self.init_weight_and_state()

    def init_weight_and_state(self):
        # load network and state
        if not self.isTrain or self.opt.continue_train:
            weight_path = os.path.join(self.opt.checkpoints_dir, 'model', 'best.pt')
            nets = torch.load(weight_path, map_location=self.device)
            netG_state_dict, netD_state_dict = nets['netG_state_dict'], nets['netD_state_dict']
            self.total_iter = nets['total_iter']
            self.epoch_past = nets['epoch']
            self.netG.load_state_dict(netG_state_dict)
            self.netD.load_state_dict(netD_state_dict)

            self.netG.to(self.device)
            self.netD.to(self.device)

        # print networks
        num_params_G = 0
        for param in self.netG.parameters():
            num_params_G += param.numel()
        self.logger.info('[Network %s] Total number of parameters : %.3f M' % ('G', num_params_G / 1e6))
        num_params_D = 0
        for param in self.netD.parameters():
            num_params_D += param.numel()
        self.logger.info('[Network %s] Total number of parameters : %.3f M' % ('D', num_params_D / 1e6))

    def save_networks(self, epoch=0, total_iter=0, best_model=False):
        model_dict = {
            'epoch': epoch,
            'total_iter': total_iter,
            # 'netG_state_dict': self.netG.to(torch.device('cpu')).state_dict(),
            'netG_state_dict': self.netG.state_dict(),
            # 'netG_state_dict': self.netG.to(torch.device('cpu')).state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            # 'optimizerG': self.optimizerG.state_dict(),
            # 'optimizerD': self.optimizerD.state_dict()
        }
        if best_model:
            savepath = os.path.join(self.opt.model_dir, 'best_model.pt')
        else:
            savepath = os.path.join(self.opt.model_dir, '{}.pt'.format(total_iter))
        torch.save(model_dict, savepath)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
        lr = self.optimizers[0].param_groups[0]['lr']
        self.logger.info('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.fake_B.shape != self.real_A.shape:
            img_H, img_W = self.real_A.shape[-2], self.real_A.shape[-1]
            self.fake_B = F.interpolate(self.fake_B, size=(img_H, img_W), mode='bilinear', align_corners=True)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)

        self.loss_D = self.criterionD(pred_real, pred_fake)
        # combine loss and calculate gradients
        self.loss_D.backward()

        self.writer.add_scalar('loss_D', self.loss_D.item(), self.total_iter)

        self.logger.info('-----------iter {}:-------------'.format(self.total_iter))
        self.logger.info('  loss_D   : {} '.format(self.loss_D.item()))

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.fake_B.shape != self.real_A.shape:
            img_H, img_W = self.real_A.shape[-2], self.real_A.shape[-1]
            self.fake_B = F.interpolate(self.fake_B, size=(img_H, img_W), mode='bilinear', align_corners=True)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionG(pred_fake)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

        self.writer.add_scalar('loss_G_GAN', self.loss_G_GAN.item(), self.total_iter)
        self.writer.add_scalar('loss_G_L1', self.loss_G_L1.item(), self.total_iter)

        self.logger.info('  loss_G_GAN   : {} '.format(self.loss_G_GAN.item()))
        self.logger.info('  loss_G_L1   : {} '.format(self.loss_G_L1.item()))

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def save_image(self, img_source, fake, total_iter, idx):
        long_img = vutils.make_grid(torch.cat([img_source, fake, fake - img_source], dim=0),
                                    padding=5, normalize=True).detach().cpu().numpy()
        long_img = (np.transpose(long_img, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)
        save_path = os.path.join(self.opt.image_dir, '{}_{}.jpg'.format(total_iter, idx))
        cv2.imwrite(save_path, long_img)

    def validation(self, total_iter):
        error = 0
        self.netD.eval()
        self.netG.eval()
        with torch.no_grad():
            for idx, (img_source, img_target) in enumerate(self.dataset_val):
                if len(img_source.shape) != 4 and len(img_target.shape) != 4:
                    continue
                if any([img_source.shape[-1] < 300, img_source.shape[-2] < 300]):
                    continue
                # print(img_source.shape)
                img_source, img_target = img_source.to(self.device), img_target.to(self.device)

                fake = self.netG(img_source)
                if fake.shape != img_source.shape:
                    img_H, img_W = img_source.shape[-2], img_source.shape[-1]
                    fake = F.interpolate(fake, size=(img_H, img_W), mode='bilinear', align_corners=True)

                if idx < 5:
                    self.save_image(img_source, fake,  total_iter, idx)
                error += self.criterionL1(fake, img_target).detach().cpu().item()
                if idx > 1000:
                    break
        error = error / 1000
        # error = error/self.validation_data_len

        self.netD.train()
        self.netG.train()
        return error

    def training(self):
        min_error = 0
        for epoch in range(self.epoch_past, self.opt.epoch):

            for iter, (image_source, image_target) in enumerate(self.dataset_train):
                if len(image_source.shape) != 4 and len(image_target.shape) != 4:
                    continue
                if any([image_source.shape[-1]<300, image_source.shape[-2]<300]):
                    continue

                self.real_A = image_source.to(self.device)
                self.real_B = image_target.to(self.device)
                self.optimize_parameters()

                if self.total_iter % self.opt.snapshot == 0:
                    self.logger.info('-----After epoch={}, iter={}, doing validation-------'.
                                     format(epoch, self.total_iter))
                    error = self.validation(self.total_iter)
                    # error = 100
                    self.logger.info('Done, error={}'.format(error))
                    self.writer.add_scalar('error', error, self.total_iter)

                    self.save_networks(epoch, self.total_iter, best_model=False)

                    if error <= min_error:
                        self.logger.info('Saving model in {}'.format(self.opt.model_dir))
                        min_error = error
                        self.save_networks(epoch, self.total_iter, best_model=True)
                self.total_iter += 1
            self.epoch_past += 1
            self.update_learning_rate()
