import os
import pandas as pd
import time
import PIL
import numpy as np
from PIL import Image

# 4. Prepare images and labels
# columnNames = list()
# columnNames.append('image_path')
# columnNames.append('label')
# for i in range(784):
#     pixel = str(i)
#     columnNames.append(pixel)

# train_data = pd.DataFrame(columns=columnNames)
# start_time = time.time()
# num_images = 0
# bad_imgs = []
# j = 0
# # for fold in thousand_outfit_list:
# for fold in tqdm(all_outfit_list):
#     print('Start outfit: ', fold)
#     for file in os.listdir(IMG_DIR + '/' + fold):
#         try:
#             if 'outfit' not in file and 'jpg' in file:
#                 print('Image: ', file)
#                 img_path = IMG_DIR+'/'+fold+'/'+file
#                 img = Image.open(os.path.join(img_path)).convert('L')
#                 img = img.resize((28,28), Image.NEAREST)
#                 pixels = list(img.getdata())
#                 imgdata = np.array(pixels)
#                 imgdata = imgdata.astype(int)
                # data = []
                # data.append(file)
                # data.append(image_to_label[file])
                # for y in range(len(imgdata)):
                #     data.append(imgdata[y])
                # train_data.loc[num_images] = data
#                 num_images += 1
#         except (PIL.UnidentifiedImageError):
#             bad_imgs.append(img_path)
#             pass
# train_data.to_csv('../jsons/image_data.csv', index=False)
# print('Created CSV file!', num_images)
# write_json('../jsons/bad_images.json', bad_imgs)