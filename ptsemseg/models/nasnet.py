import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable

from ptsemseg import caffe_pb2
from ptsemseg.models.utils import *
from ptsemseg.loss import *
from ptsemseg.utils import convert_state_dict
from ptsemseg.models.nasmobile import *

import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter

pspnet_specs = {
    'pascalvoc': 
    {
         'n_classes': 21,
         'input_size': (473, 473),
         'block_config': [3, 4, 23, 3],
    },

    'cityscapes': 
    {
         'n_classes': 3,
         'input_size': (713, 713),
         'block_config': [3, 4, 23, 3],
    },

    'ade20k': 
    {
         'n_classes': 150,
         'input_size': (473, 473),
         'block_config': [3, 4, 6, 3],
    },
}

class nasnet(nn.Module):
    
    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105

    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow

    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928

    """

    def __init__(self, 
                 n_classes=3, 
                 block_config=[3, 4, 23, 3], 
                 input_size=(713, 713), 
                 version=None):

        super(nasnet, self).__init__()
        
        self.block_config = pspnet_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = pspnet_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = pspnet_specs[version]['input_size'] if version is not None else input_size
        
        self.stem_filters = 32
        self.penultimate_filters = 1056
        self.filters_multiplier = 2

        filters = self.penultimate_filters // 24
        # 24 is default value for the architecture

        self.conv0 = nn.Sequential()
        # S/2
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2,
                                                bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))

        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // (self.filters_multiplier ** 2))
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // self.filters_multiplier)

        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters//2, # 1, 0.5
                                in_channels_right=2*filters, out_channels_right=filters) # 2, 1
        self.cell_1 = NormalCell(in_channels_left=2*filters, out_channels_left=filters, # 2, 1
                                 in_channels_right=6*filters, out_channels_right=filters) # 6, 1
        self.cell_2 = NormalCell(in_channels_left=6*filters, out_channels_left=filters, # 6, 1
                                 in_channels_right=6*filters, out_channels_right=filters) # 6, 1
        self.cell_3 = NormalCell(in_channels_left=6*filters, out_channels_left=filters, # 6, 1
                                 in_channels_right=6*filters, out_channels_right=filters) # 6, 1

        self.reduction_cell_0 = ReductionCell0(in_channels_left=6*filters, out_channels_left=2*filters, # 6, 2
                                               in_channels_right=6*filters, out_channels_right=2*filters) # 6, 2

        self.cell_6 = FirstCell(in_channels_left=6*filters, out_channels_left=filters, # 6, 1
                                in_channels_right=8*filters, out_channels_right=2*filters) # 8, 2
        self.cell_7 = NormalCell(in_channels_left=8*filters, out_channels_left=2*filters, # 8, 2
                                 in_channels_right=12*filters, out_channels_right=2*filters) # 12, 2
        self.cell_8 = NormalCell(in_channels_left=12*filters, out_channels_left=2*filters, # 12, 2
                                 in_channels_right=12*filters, out_channels_right=2*filters) # 12, 2
        self.cell_9 = NormalCell(in_channels_left=12*filters, out_channels_left=2*filters, # 12, 2
                                 in_channels_right=12*filters, out_channels_right=2*filters) # 12, 2

        self.reduction_cell_1 = ReductionCell1(in_channels_left=12*filters, out_channels_left=4*filters, # 12, 4
                                               in_channels_right=12*filters, out_channels_right=4*filters) # 12, 4

        self.cell_12 = FirstCell(in_channels_left=12*filters, out_channels_left=2*filters, # 12, 2
                                 in_channels_right=16*filters, out_channels_right=4*filters) # 16, 4
        self.cell_13 = NormalCell(in_channels_left=16*filters, out_channels_left=4*filters, # 16, 4
                                  in_channels_right=24*filters, out_channels_right=4*filters) # 24, 4
        self.cell_14 = NormalCell(in_channels_left=24*filters, out_channels_left=4*filters, # 24, 4
                                  in_channels_right=24*filters, out_channels_right=4*filters) # 24, 4
        self.cell_15 = NormalCell(in_channels_left=24*filters, out_channels_left=4*filters, # 24, 4
                                  in_channels_right=24*filters, out_channels_right=4*filters) # 24, 4

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(1056, [6, 3, 2, 1])
       
        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(2112, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.convbnrelu4_aux = conv2DBatchNormRelu(in_channels=528, k_size=3, n_filters=256, padding=1, stride=1, bias=False)
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        inp_shape = x.shape[2:]

        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)
        # Size / 16
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        # Size / 32
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        
        # Auxiliary layers for training
        x_aux = self.convbnrelu4_aux(x_cell_9)
        x_aux = self.dropout(x_aux)
        x_aux = self.aux_cls(x_aux)

       
        x = self.pyramid_pooling(x_cell_15)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear')

        if self.training:
            return x_aux, x
        else: # eval mode
            return x

    def load_init_model(self, path=None):
        own_state = self.state_dict()
        if path:
            state_dict = torch.load(path)['model_state']
        else:
            state_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/nasnetamobile-7e03cead.pth')
        for n, p in state_dict.items():
            if path:
                n = n[7:]
            if n not in own_state:
                continue
            if isinstance(p, Parameter):
                p = p.data
            print(n)
            own_state[n].copy_(p)

    def load_pretrained_model(self, path=None):
        own_state = self.state_dict()
        state_dict = torch.load(path)['model_state']
        for name, param in state_dict.items():
            if name[7:] not in own_state.keys():
                continue
            if isinstance(param, Parameter):
                param = param.data
            print(name)
            own_state[name[7:]].copy_(param)        


# For Testing Purposes only
if __name__ == '__main__':
    cd = 0
    import os
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import scipy.misc as m
    from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cl
    psp = pspnet(version='cityscapes')
    own_state = psp.state_dict()
    state_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/nasnetamobile-7e03cead.pth')
    for n, p in state_dict.items():
        if n not in own_state:
            continue
        if isinstance(p, Parameter):
            p = p.data
        print(n)
        own_state[n].copy_(p)
 
    #psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet50_ADE20K.caffemodel'))
    #psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet101_VOC2012.caffemodel'))
    
    # psp.load_state_dict(torch.load('psp.pth'))

    psp.float()
    psp.cuda(cd)
    psp.eval()

    #dataset_root_dir = '/home/dannyhung/LightNet/datasets/cityscapes/'
    #dst = cl(root=dataset_root_dir)
    #img = m.imread(os.path.join(dataset_root_dir, 'leftImg8bit/train/aachen/aachen_000012_000019_leftImg8bit.png'))
    img = m.imread('carla/leftImg8bit/train/135.png')
    m.imsave('cropped.png', img)
    orig_size = img.shape[:-1]
    img = m.imresize(img, (320, 416))
    img = img[:, :, ::-1]
    #img = img.transpose(2, 0, 1)
    img = img.astype(np.float64)
    img -= np.array([92.05991654, 84.79349942, 77.08157727])[None, None, :]
    img /= 255.
    #img = np.copy(img[::-1, :, :])
    img = img.transpose(2, 0 ,1)
    img = torch.from_numpy(img).float() # convert to torch tensor
    img = Variable(img.unsqueeze(0)).cuda()

    out = psp(img)
    pred = out.data.cpu().max(1)[1].numpy()

    from collections import Counter
    print(pred.shape)
    #img = pred[:, :]
    #l = []
    #for i in img:
    #    for j in i:
    #        l.append(j)
    #print(Counter(l))
    #decoded = dst.decode_segmap(pred)
    #m.imsave('cityscapes_berlin_tiled.png', decoded)
    ##m.imsave('cityscapes_sttutgart_tiled.png', pred) 

    #checkpoints_dir_path = 'checkpoints'
    #if not os.path.exists(checkpoints_dir_path):
    #    os.mkdir(checkpoints_dir_path)
    #psp = torch.nn.DataParallel(psp, device_ids=range(torch.cuda.device_count())) # append `module.`
    #state = {'model_state': psp.state_dict()}
    #torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_cityscapes.pth"))
    ##torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_50_ade20k.pth"))
    ##torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_pascalvoc.pth"))
    #print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))

