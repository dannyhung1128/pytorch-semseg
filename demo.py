import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

import sys, os
import time
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from collections import Counter

# prepare model
model = get_model('pspnet', 3, version='carla')
state = convert_state_dict(torch.load('checkpoints/pspnet_carla_10_best_model.pkl')['model_state'])
model.load_state_dict(state)
model.eval()
model.cuda()


#file = sys.argv[-1]
file = 'test_video.mp4'
video = skvideo.io.vread(file)

answer_key = {}

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

# Frame numbering starts at 1
frame = 1

for rgb_frame in video:
    start = time.time()
    img = rgb_frame[:, :, ::-1]
    img = img.astype(np.float64)
    img -= np.array([92.05991654, 84.79349942, 77.08157727])[None, None, :]
    img /= 255.
    img = img.transpose(2, 0 ,1)
    img = torch.from_numpy(img).float() # convert to torch tensor
    img = Variable(img.unsqueeze(0)).cuda()

    out = model(img)
    pred = np.argmax(out, axis=1)[0]
    
    # Grab prediction
    #outputs = model(rgb_frame)
    #pred = outputs.data.max(1)[1].cpu().numpy()
    # Look for red cars :)
    binary_car_result = pred.copy()
    #binary_car_result[binary_car_result == 1] = -1
    binary_car_result[binary_car_result != 1] = 0
    #binary_car_result[binary_car_result == -1] = 1
    # Look for road :)
    binary_road_result = pred.copy()
    binary_road_result[binary_car_result == 0] = -1
    binary_road_result[binary_car_result != -1] = 0
    binary_road_result[binary_car_result == -1] = 1
    
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    duration = time.time() - start
    # Increment frame
    frame+=1
    print(frame, duration)

# Print output in proper json format
print (json.dumps(answer_key))

