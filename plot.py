from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
import cv2

data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

# Setup Dataloader
data_loader = get_loader("cityscapes")
data_path = get_data_path("cityscapes")
t_loader = data_loader(data_path, is_transform=True, img_size=(1024, 2048))

label = t_loader[1][1]
print(label.shape)
label = np.array(label)
l = []
for i in range(1024):
	for j in range(2048):
		l.append(label[i][j])
from collections import Counter
c = Counter(l)
print(c)
print(t_loader.files[t_loader.split][1].rstrip())
cv2.imwrite('test.png', np.array(label))

print(t_loader[1][0].shape)