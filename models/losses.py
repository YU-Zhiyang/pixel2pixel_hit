import torch
import torch.nn as nn
import torch.nn.functional as F


class DisLoss(object):
    __slots__ = ['weight', 'gan_mode']
    def __init__(self, gan_mode='lsgan', weight=1):

        super(DisLoss, self).__init__()
        self.weight = weight
        self.gan_mode = gan_mode

    def __call__(self, real, fake):

        if self.gan_mode == 'lsgan':
            real_lable = torch.ones_like(real, requires_grad=False)
            fake_lable = torch.zeros_like(fake, requires_grad=False)
            loss = self.weight * (F.mse_loss(real, real_lable) + F.mse_loss(fake, fake_lable))*0.5
            return loss

        elif self.gan_mode == 'wgangp':
            loss = self.weight * (fake.mean() - real.mean())
            return loss
        elif self.gan_mode == 'hinge':
            loss = self.weight * (F.relu(1.-real).mean() + F.relu(1.+fake).mean())*0.5
            return loss
        elif self.gan_mode == 'vanilla':
            real_lable = torch.ones_like(real, requires_grad=False)
            fake_lable = torch.zeros_like(fake, requires_grad=False)
            loss = self.weight * (F.binary_cross_entropy_with_logits(fake, fake_lable) +
                                  F.binary_cross_entropy_with_logits(real, real_lable))*0.5
            return loss


class GanLoss(object):
    __slots__ = ['weight', 'gan_mode']

    def __init__(self, gan_mode='lsgan', weight=1):

        super(GanLoss, self).__init__()
        self.weight = weight
        self.gan_mode = gan_mode

    def __call__(self, fake):

        if self.gan_mode == 'lsgan':
            fake_lable = torch.ones_like(fake, requires_grad=False)
            loss = F.mse_loss(fake, fake_lable)
            return loss

        elif self.gan_mode == 'wgangp':
            loss = -self.weight * fake.mean()
            return loss
        elif self.gan_mode == 'hinge':
            loss = -self.weight * fake.mean()
            return loss
        elif self.gan_mode == 'vanilla':
            fake_lable = torch.ones_like(fake, requires_grad=False)
            loss = F.binary_cross_entropy_with_logits(fake, fake_lable)
            return loss


class L1Loss(object):
    __slots__ = ['weight']

    def __init__(self, weight):
        super(L1Loss, self).__init__()
        self.weight = weight

    def __call__(self, source, target, mask=None):
        if mask is None:
            return self.weight * F.l1_loss(source, target)
        else:
            mask.requires_grad = False
            total_pixel_num = mask.numel() * 1.0
            pos_pixel_idx = mask.ge(0.5)
            pos_pixel_num = pos_pixel_idx.sum().item() * 1.0
            mask[pos_pixel_idx] = (total_pixel_num - pos_pixel_num) / total_pixel_num
            mask[1-pos_pixel_idx] = pos_pixel_num/total_pixel_num
            mask = mask.expand_as(source)

            return self.weight * (F.l1_loss(source * mask, target * mask))


class L2loss(object):
    __slots__ = ['weight']

    def __init__(self, weight):
        super(L2loss, self).__init__()
        self.weight = weight

    def __call__(self, source, target, mask=None):
        if mask is None:
            return self.weight * F.mse_loss(source, target)
        else:
            mask.requires_grad = False
            total_pixel_num = mask.numel() * 1.0
            pos_pixel_idx = mask.ge(0.5)
            pos_pixel_num = pos_pixel_idx.sum().item() * 1.0
            mask[pos_pixel_idx] = (total_pixel_num - pos_pixel_num) / total_pixel_num
            mask[1 - pos_pixel_idx] = pos_pixel_num / total_pixel_num
            mask = mask.expand_as(source)

            return self.weight * (F.mse_loss(source * mask, target * mask))


class TVloss(object):
    __slots__ = ['weight']

    def __init__(self, weight=1):
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size




if __name__ == '__main__':
    a = torch.rand((128, 3, 7, 7))
    b = torch.ones((128, 3, 7, 7))
    loss = L1Loss(weight=1)
    mask = torch.rand((7,7))

    c = loss(a, b, mask)



