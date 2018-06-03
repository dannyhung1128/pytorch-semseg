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
__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides,bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class pspnetXception(nn.Module):
    
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

        super(pspnetXception, self).__init__()
        
        self.block_config = pspnet_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = pspnet_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = pspnet_specs[version]['input_size'] if version is not None else input_size
        
        self.conv1 = nn.Conv2d(3, 32, 3,2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        # Size/2
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        # Size/4
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        # Size/8
        self.block3=Block(256,728,2,1,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        # Size /2
        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
        
        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])
       
        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.convbnrelu4_aux = conv2DBatchNormRelu(in_channels=1536, k_size=3, n_filters=256, padding=1, stride=1, bias=False)
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        inp_shape = x.shape[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #print(x.shape)
        
        # Auxiliary layers for training
        x_aux = self.convbnrelu4_aux(x)
        x_aux = self.dropout(x_aux)
        x_aux = self.aux_cls(x_aux)
        
        x = self.conv4(x)
        x = self.bn4(x)
        #print(x.shape)
        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear')

        if self.training:
            return x_aux, x
        else: # eval mode
            return x


    

def get_pretrained_model():
    settings = pretrained_settings['xception']['imagenet']
    
    model = pspnetXception()
    own_state = model.state_dict()
    state_dict = model_zoo.load_url(settings['url'])
    for n, p in state_dict.items():
        if n not in own_state or 'fc' in n:
            continue
        #if 'pointwise' in n:
        #    p = p.data
        #    p = p.unsqueeze(-1)
        #    p = p.unsqueeze(-1)
        if isinstance(p, Parameter):
            p = p.data
        own_state[n].copy_(p)


    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model



# For Testing Purposes only
if __name__ == '__main__':
    cd = 0
    import os
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import scipy.misc as m
    from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cl
    psp = get_pretrained_model()
    
    # Just need to do this one time
    #caffemodel_dir_path = '/home/dannyhung/pytorch-semseg/'
    #psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet101_cityscapes.caffemodel'))
    
    #psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet50_ADE20K.caffemodel'))
    #psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet101_VOC2012.caffemodel'))
    
    # psp.load_state_dict(torch.load('psp.pth'))

    psp.float()
    #psp.cuda(cd)
    psp.train()

    #dataset_root_dir = '/home/dannyhung/LightNet/datasets/cityscapes/'
    #dst = cl(root=dataset_root_dir)
    #img = m.imread(os.path.join(dataset_root_dir, 'leftImg8bit/train/aachen/aachen_000012_000019_leftImg8bit.png'))
    img = m.imread('carla/leftImg8bit/train/135.png')
    m.imsave('cropped.png', img)
    orig_size = img.shape[:-1]
    img = m.imresize(img, (180, 240))
    img = img[:, :, ::-1]
    #img = img.transpose(2, 0, 1)
    img = img.astype(np.float64)
    img -= np.array([92.05991654, 84.79349942, 77.08157727])[None, None, :]
    img /= 255.
    #img = np.copy(img[::-1, :, :])
    img = img.transpose(2, 0 ,1)
    img = torch.from_numpy(img).float() # convert to torch tensor
    img = img.unsqueeze(0)

    out = psp(img)
    pred = out[0].data.max(1)[1].numpy()

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
#
    #checkpoints_dir_path = 'checkpoints'
    #if not os.path.exists(checkpoints_dir_path):
    #    os.mkdir(checkpoints_dir_path)
    #psp = torch.nn.DataParallel(psp, device_ids=range(torch.cuda.device_count())) # append `module.`
    #state = {'model_state': psp.state_dict()}
    #torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_cityscapes.pth"))
    ##torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_50_ade20k.pth"))
    ##torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_pascalvoc.pth"))
    #print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))

