import torch
import torch.nn.functional as F
__all__ = ['SNGenLoss', 'SNDisLoss', 'TVLoss', 'L1ReconLoss']


class SNGenLoss(object):
    """
    The hinge loss for generator
    """
    __slots__ = ['weight']

    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def __call__(self, neg):
        return - self.weight * torch.mean(neg)
        # return - self.weight * torch.sum(torch.mean(neg, dim=[-1, -2]))


class SNDisLoss(object):
    """
    The  hinge loss for discriminator
    """
    __slots__ = ['weight']

    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def __call__(self, real, fake):

        loss = self.weight * torch.sum(torch.mean(F.relu(1.-real), dim=[-1, -2]) +
                                       torch.mean(F.relu(1.+fake), dim=[-1, -2]))
        return loss


class TVLoss(object):
    """
    Total Variation loss
    """
    __slots__ = ['weight']

    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def __call__(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class L1ReconLoss(object):
    """
    L1 Reconstruction loss for two image
    """
    __slots__ = ['weight']

    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def __call__(self, imgs, recon_imgs, masks=None):
        if masks is None:
            # return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
            return self.weight * F.l1_loss(imgs, recon_imgs)
        else:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1,1,1,1))


class L2ReconLoss(object):
    """
    L1 Reconstruction loss for two image
    """
    __slots__ = ['weight']

    def __init__(self, weight=1):
        super(L2ReconLoss, self).__init__()
        self.weight = weight

    def __call__(self, imgs, recon_imgs, masks=None):
        if masks is None:
            # return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
            return self.weight * F.mse_loss(imgs, recon_imgs)
        else:
            return self.weight * torch.mean((imgs - recon_imgs).pow(2) / masks.view(masks.size(0), -1).mean(1).view(-1,1,1,1))


class L1(object):
    __slots__ = ['weight']

    def __init__(self, weight=1):
        super(L1, self).__init__()
        self.weight = weight

    def __call__(self, imgs, masks=None):
        if masks is None:
            # return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
            return self.weight * torch.mean(torch.abs(imgs))
        else:
            return self.weight * torch.mean(
                torch.abs(imgs) / masks.view(masks.size(0), -1).mean(1).view(-1, 1, 1, 1))


if __name__ == '__main__':
    loss = TVLoss()
    img = torch.rand((1,3,255,255))
    tvloss = loss(img)
    pass