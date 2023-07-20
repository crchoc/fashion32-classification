import json
import sys
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io
from PIL import Image
import time
import pickle as pkl

with open('../jsons/label_data.json') as t:
    data = json.load(t)

data_for_cnn = {}
for outfit in data.values():
    for item in outfit:
        for label in outfit[item]['label']:
            if len(outfit[item]['label']['98']) > 0:
                data_for_cnn[item] = outfit[item]['label']['98'][0]
            if len(outfit[item]['label']['99']) > 0:
                data_for_cnn[item] = outfit[item]['label']['99'][0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(), normalize
            ])

def process_image(im):
    im = transform(im)
    im = im.unsqueeze_(0)
    im = im.to(device)
    out = model(im)
    return out.squeeze()

dataset_path = '..'
images_path = dataset_path + '/images/'

save_to = '../dataset'
if not os.path.exists(save_to):
    os.makedirs(save_to)
save_dict = os.path.join(save_to, 'imgs_featdict_test.pkl')

features = {}
count = {}
i = 0
n_items = len(data_for_cnn.keys())
with torch.no_grad():
    for item in data_for_cnn:
        outfit_id, _, image_id = item.split('_')
        image_path = images_path + outfit_id + '/' + item
        im = skimage.io.imread(image_path)
        if len(im.shape)==2:
            im = gray2rgb(im)
        if im.shape[2] == 4:
            im = rgba2rgb(im)
        
        im = resize(im, (256,256))
        im = img_as_ubyte(im)
        feats = process_image(im).cpu().numpy()
        if image_id not in features:
            features[image_id] = feats
            count[image_id] = 0
        else:
            features[image_id] += feats
        count[image_id] += 1
        i += 1

feat_dict = {}
for id in features:
    feats = features[id]
    feats = np.array(feats)/count[id]
    feat_dict[id] = feats

with open(save_dict, 'wb') as handle:
    pkl.dump(feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
print('Done!')