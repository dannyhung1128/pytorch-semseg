import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [#[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [  0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

    label_colours = dict(zip(range(3), colors))

    mean_rgb = {'pascal': [103.939, 116.779, 123.68], 'cityscapes': [66.63619917, 76.12220271, 65.74310554]}#[73.15835921, 82.90891754, 72.39239876]} # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(512, 1024), augmentations=None, img_norm=True, version='cityscapes', arch='pspnet'):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 3
        self.arch = arch
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb['cityscapes'])
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')
        #self.files[split] = self.files[split][:50]
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.bg_classes   = [8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 31, 32, 33]
#        self.void_classes += [8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 31, 32, 33]
                        
        self.valid_classes = [7, 26, 99] # 26, 27, 28 = vehicle
        self.class_names = ['road', 'vehicle', 'background']
        #self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
        #                    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
        #                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
        #                    'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(3))) 
        self.n_pad = int(self.img_size[1] * 0.75 - self.img_size[0])
        self.l_bound = int(self.n_pad * 29 / 80)
        self.padding = np.zeros((3, self.n_pad, self.img_size[1]))
        self.padding_lbl = np.ones((self.n_pad, self.img_size[1])) * 250
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2], 
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        #img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)

        #if not np.all(classes == np.unique(lbl)):
        #    print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
            print('after det', classes,  np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        img = np.concatenate((self.padding[:, self.l_bound:, :], img, self.padding[:, :self.l_bound, :]), axis=1)
        lbl = np.concatenate((self.padding_lbl[self.l_bound:], lbl, self.padding_lbl[:self.l_bound]), axis=0)
        if self.arch == 'pspnet':      
            img = img[:, 62:-22 ,:]
            lbl = lbl[62:-22, :]
        #img = img[:, 74:-38 ,:]
        #lbl = lbl[74:-38, :]        
        # deeplab 224x448
        elif self.arch == 'deeplabv3':
            img = img[:, 106:-38, :]
            lbl = lbl[106:-38, :]
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
        return rgb

    def encode_segmap(self, mask):
        #Put all void classes to zero
        mask[mask == 27] = 26 # truck to car
        mask[mask == 28] = 26 # bus to car
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _bgc in self.bg_classes:
            mask[mask==_bgc] = self.valid_classes[-1]
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048),
                             RandomRotate(10),
                             RandomHorizontallyFlip()])

    local_path = '/home/meetshah1995/datasets/cityscapes/'
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs,2)
        for j in range(bs):      
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()
