import sys, os
import time
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.lovasz import *
from ptsemseg.augmentations import *
from ptsemseg.utils import convert_state_dict
from ptsemseg.models.pspnetXception import get_pretrained_model

def train(args):

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    if args.dataset == 'carla':
        v_loader = data_loader(data_path, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)
    else:
        v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)
    print(len(trainloader))
    # Setup Metrics
    running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes)
    
    if args.arch == 'pspnet':
        if args.dataset == 'cityscapes':
            model_path = 'checkpoints/pspnet_180x240_carla_best_model.pkl'
            model.load_pretrained_model(model_path)
            #model_path = 'pspnet_cityscapes_27_best_model.pkl'
            #state = convert_state_dict(torch.load(model_path)['model_state'])
            #model.load_state_dict(state)
            print("Loading model from ", model_path)
            #caffemodel_dir_path = '/home/dannyhung/pytorch-semseg/'
            #model.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet101_cityscapes.caffemodel'))
        elif args.dataset == 'carla':
            model_path = 'pspnet_carla_8_best_model.pkl'
            state = convert_state_dict(torch.load(model_path)['model_state'])
            model.load_state_dict(state)     
            print("Loading model from ", model_path)
    elif args.arch == 'deeplabv3plus':
        #state_dict = torch.load('/home/dannyhung/deeplab-pytorch/data/models/deeplab_resnet101/coco_init/deeplabv3plus_resnet101_COCO_init.pth')
        state_dict = convert_state_dict(torch.load('deeplabv3plus_cityscapes_20_best_model.pkl'))
        model.load_state_dict(state_dict, strict=False)   
    elif args.arch == 'icnet':
        state_dict = convert_state_dict(torch.load('/home/dannyhung/pytorch-semseg/icnet_cityscapes_9_best_model.pkl'))
        model.load_state_dict(state_dict, strict=False)
        #caffemodel_dir_path = 'checkpoints/icnet_cityscapes_trainval_90k.caffemodel'
        #model.load_pretrained_model(model_path=caffemodel_dir_path)
    elif args.arch == 'pspnetXception':
        model = get_pretrained_model()
        #state_dict = convert_state_dict(torch.load('pspnetXception_cityscapes_20_best_model.pkl'))
        #model.load_state_dict(state_dict)
        print("Loading pretrained model")
    elif args.arch == 'nasnet':
        model.load_pretrained_model('nasnet_cityscapes_29_best_model.pkl')
        #state_dict = convert_state_dict(torch.load('nasnet_cityscapes_21_best_model.pkl'))
        #model.load_state_dict(state_dict)        
        print("NASNET loaded")
    elif args.arch == 'deeplabv3':
        model.scale.mobilenetv2 = torch.nn.DataParallel(model.scale.mobilenetv2).cuda()
        #state_dict = torch.load('deeplabv3_cityscapes_24_best_model.pkl')['model_state']
        model.scale.load_pretrained_model('deeplabv3_cityscapes_28_best_model.pkl')
        #model.scale.loadMobileNetv2('mobilenetv2_718.pth.tar')
        print('mobilenetV2 loaded')

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d
    if args.loss == 'lovasz':
        print("Using Lovasz loss")
    loss_fn_1 = model.module.loss
    loss_fn_2 = multi_scale_lovasz

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
#           print(images.shape, labels.shape)   
#           if i == 2: break
            if len(images) == 1:
                images = torch.cat((images, images), 0)
                labels = torch.cat((labels, labels), 0)
            start = time.time()
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            
            optimizer.zero_grad()
            outputs = model(images)
#            for i in range(len(outputs)):
#                print(outputs[i].shape)
            forward_time = time.time() - start
            start = time.time()
            loss = loss_fn_1(input=outputs, target=labels)
            loss+= loss_fn_2(outputs, labels, ignore=250)
            loss.backward()
            optimizer.step()
            backprop_time = time.time() - start
            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d], Iter [%i/%i], Loss: %.4f, Forward time: %.4f, Backprop time: %.4f" % \
                        (epoch+1, args.n_epoch, i, len(trainloader), loss.data[0], forward_time, backprop_time))

        model.eval()
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)
            outputs = model(images_val)
    #        print(outputs.shape)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}_{}_{}_best_model.pkl".format(args.arch, args.dataset, str(epoch+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.add_argument('--loss', type=str, default='multi_ce',
                        help='Losvasz Loss or multiclass cross entropy')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    train(args)
