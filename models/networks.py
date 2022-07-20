import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision.models as torch_models
from .warmup_scheduler import GradualWarmupScheduler


###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'warmup':
        cur_lr_policy = opt.lr_policy_after
    else:
        cur_lr_policy = opt.lr_policy

    if cur_lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cur_lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif cur_lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif cur_lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    if opt.lr_policy == 'warmup':
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.n_epochs_warmup, after_scheduler=scheduler)

    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_iern(n_emo=6, n_cnfnd=3, feat_nc=512, feat_size=7, n_gen_blk=2, n_recons_blk=3, n_dis_blk=3,
                  backbone_archi='resnet50', pretrained=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if 'resnet' in backbone_archi:
        backbone = getattr(torch_models, backbone_archi)(pretrained=pretrained)
        # remove classifier and update the last layer channel to 512
        feat_list = list(backbone.children())[:-2]
        feat_list += [nn.Conv2d(2048, feat_nc, kernel_size=1, stride=1)]
        backbone = nn.Sequential(*feat_list)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            backbone.to(gpu_ids[0])
            backbone = torch.nn.DataParallel(backbone, gpu_ids)

    emo_gen = init_net(FeatEncoderNet(feat_nc, n_gen_blk), init_type, init_gain, gpu_ids)
    con_gen = init_net(FeatEncoderNet(feat_nc, n_gen_blk), init_type, init_gain, gpu_ids)
    recons_net = init_net(FeatDecoderNet(feat_nc * 2, n_recons_blk), init_type, init_gain, gpu_ids)

    emo_dis = init_net(FeatDisNet(feat_nc, n_emo, feat_size, n_dis_blk), init_type, init_gain, gpu_ids)
    con_dis = init_net(FeatDisNet(feat_nc, n_cnfnd, feat_size, n_dis_blk), init_type, init_gain, gpu_ids)

    cnfnd_net = ConfounderNet(feat_nc, feat_size, n_cnfnd)
    # put the network on 1 gpu only (if available), do not distribute confounder memory tensors.
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        cnfnd_net.to(gpu_ids[0])

    classifier = init_net(ClassifierNet(n_emo, feat_nc), init_type, init_gain, gpu_ids)

    return backbone, emo_gen, con_gen, recons_net, emo_dis, con_dis, cnfnd_net, classifier


def define_centerloss(cnfnd_num, feat_dim, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = CenterLoss(cnfnd_num, feat_dim)
    net = init_net(net, init_type, init_gain, gpu_ids)
    return net


##############################################################################
# Classes
##############################################################################


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class FeatEncoderNet(nn.Module):
    """docstring for FeatEncoderNet"""

    def __init__(self, in_channels, n_blocks=2, norm_layer=nn.BatchNorm2d):
        super(FeatEncoderNet, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)]
        for idx in range(n_blocks):
            layers += [ResnetBlock(in_channels, 'zero', norm_layer, False, use_bias)]

        self.feat_encoder = nn.Sequential(*layers)

    def forward(self, x):
        out_feat = self.feat_encoder(x)

        return out_feat


class FeatDecoderNet(nn.Module):
    """docstring for FeatDecoderNet"""

    def __init__(self, in_channels, n_blocks=3, norm_layer=nn.BatchNorm2d):
        super(FeatDecoderNet, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        out_channels = int(in_channels / 2)
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        for idx in range(n_blocks):
            layers += [ResnetBlock(out_channels, 'zero', norm_layer, False, use_bias)]

        self.feat_decoder = nn.Sequential(*layers)

    def forward(self, x1, x2):
        combined_x = torch.cat((x1, x2), 1)
        out_feat = self.feat_decoder(combined_x)

        return out_feat


class FeatDisNet(nn.Module):
    """docstring for FeatDisNet"""

    def __init__(self, in_channels=512, n_emo=6, feat_size=7, repeat_num=3):
        super(FeatDisNet, self).__init__()
        layers = []

        layers += [nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.01)]

        curr_dim = in_channels
        for i in range(repeat_num):
            layers += [nn.Conv2d(int(curr_dim), int(curr_dim / 2), kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(0.01)]
            curr_dim = int(curr_dim / 2)

        kernel_size = int(feat_size / 2)
        layers += [nn.Conv2d(curr_dim, n_emo, kernel_size=kernel_size, bias=False)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        pred_expr = self.model(x)
        pred_expr = torch.squeeze(pred_expr)
        return pred_expr


class ConfounderNet(nn.Module):
    """docstring for ConfounderNet"""

    def __init__(self, in_channels=512, feat_size=7, confound_nc=3):
        super(ConfounderNet, self).__init__()

        self.confound_nc = confound_nc
        self.register_buffer('mem_tensors', torch.zeros([confound_nc, in_channels, feat_size, feat_size]))

    def forward(self, center_weights=None, is_test=False):
        if (not is_test) and (center_weights is not None):
            self.mem_tensors = center_weights.view(*list(self.mem_tensors.size()))

        return self.mem_tensors


class ClassifierNet(nn.Module):
    """resnet classifier part used for the baseline, following the input & output setting of ClassifierNet"""

    def __init__(self, n_emo, feat_nc=2048):
        super(ClassifierNet, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feat_nc, n_emo)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        pred_expr = self.fc(x)

        return pred_expr


class CenterLoss(nn.Module):
    """Center loss. https://github.com/KaiyangZhou/pytorch-center-loss

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_parameter("centers", nn.Parameter(torch.randn(self.num_classes, self.feat_dim)))
        self.register_buffer("classes", torch.arange(self.num_classes).long())

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # dismat = (x - centers) ^ 2
        x = x.view(batch_size, -1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # mask the corresponding class for each x
        # classes = torch.arange(self.num_classes).long()
        # if x.is_cuda:
        #     classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.clone().expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
