import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

class Cutout(object):
    def __init__(self, n_holes, length):
        """
        :param n_holes: Number of holes to cut out
        :param length: Length of the cutout square
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        :param img: Input PIL image
        :return: PIL image with cutout applied
        """
        img = np.array(img)
        h, w, _ = img.shape

        mask = np.ones((h, w), np.uint8)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0

        img = img * mask[:, :, np.newaxis]

        return transforms.ToPILImage()(img)

def mixup(x, y, xm, ym, alpha=1.0, use_cuda=True):
    """
    :param x: Input images
    :param y: Input labels
    :param alpha: MixUp interpolation coefficient (0: only main image, 1: random interpolation)
    :param use_cuda: Whether to use GPU (True) or CPU (False)
    :return: Mixed images, mixed labels, and the MixUp coefficients (lambdas)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * xm

    return mixed_x, y, ym, lam

def mixup_data(x, y, alpha=1.0):
    """
    :param x: Input images
    :param y: Input labels
    :param alpha: MixUp interpolation coefficient (0: only main image, 1: random interpolation)
    :param use_cuda: Whether to use GPU (True) or CPU (False)
    :return: Mixed images, mixed labels, and the MixUp coefficients (lambdas)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    :param criterion: Loss function (e.g., CrossEntropyLoss)
    :param pred: Model predictions
    :param y_a: Ground truth labels for the first image
    :param y_b: Ground truth labels for the second image
    :param lam: MixUp coefficients (lambdas)
    :return: Weighted loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix(x, y, alpha=1.0, use_cuda=True):
    """
    :param x: Input images
    :param y: Input labels
    :param alpha: CutMix interpolation coefficient (0: only main image, 1: random interpolation)
    :param use_cuda: Whether to use GPU (True) or CPU (False)
    :return: Mixed images, mixed labels, and the CutMix coefficients (lambdas)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    :param criterion: Loss function (e.g., CrossEntropyLoss)
    :param pred: Model predictions
    :param y_a: Ground truth labels for the first image
    :param y_b: Ground truth labels for the second image
    :param lam: CutMix coefficients (lambdas)
    :return: Weighted loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)