from PIL import Image
from numpy import asarray
import json
import numpy as np
import os
from skimage import color
from skimage import io

with open('../jsons/label_data.json') as f:
    data = json.load(f)

prep_data = {}
for outfit in data.values():
    for item in outfit:
        for label in outfit[item]['label']:
            if len(outfit[item]['label']['98']) > 0:
                prep_data[item] = outfit[item]['label']['98'][0]
            if len(outfit[item]['label']['99']) > 0:
                prep_data[item] = outfit[item]['label']['99'][0]

label_data = {}
feats_data = []
for outfit in data:
    for item in data[outfit]:
        image_path = '../images/' + outfit + '/' + item
        _, _, item_id = item.split('_')
        img = color.rgb2gray(io.imread(image_path))
        feats_data.append(asarray(img))
        label_data[item] = prep_data[item]
print(feats_data)

with open('try.json', 'w') as f:
    json.dump(feats_data, f, indent=4)
