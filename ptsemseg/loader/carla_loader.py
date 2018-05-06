import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class carlaLoader(data.Dataset):
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    label_colours = dict(zip(range(3), colors))

    mean_rgb = {'carla': [92.05991654, 84.79349942, 77.08157727]} # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(600, 800), augmentations=None, img_norm=True, version='carla'):
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
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')
    
        self.void_classes = [3]
        self.bg_classes = [0, 1, 2, 4, 5, 6, 8, 9, 11, 12]
                        
        self.valid_classes = [7, 10, 99] # 6 Roadlines -> 7
        self.class_names = ['road', 'vehicle', 'background']
        
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(3))) 

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
                                os.path.basename(img_path))
        
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)[:, :, 0]
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
        img = img[:, :, ::-1] # RGB -> BGR
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

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
            print('after det', classes,  np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

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
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        #Put all void classes to zero
        mask[mask == 6] = 7 # roadline to road
        idx = mask[490:, :] == 10
        mask[490:][idx] = self.ignore_index
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

    local_path = '/Users/dannyhung/Documents/selfDriving/pytorch-semseg/carla/'
    dst = carlaLoader(local_path, is_transform=True, augmentations=augmentations)
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

