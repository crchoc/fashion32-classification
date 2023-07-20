# import pickle as pkl
# import pandas as pd
# import json 
# import sys
# import os

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns

# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from IPython.display import Image
# from PIL import Image
# from numpy import asarray

# loadmodel = pkl.load(open('../dataset/imgs_featdict_test.pkl', 'rb'))

# with open('../jsons/label_data.json') as f:
#     data = json.load(f)

# with open('../jsons/label_types.json') as f:
#     labels = json.load(f)

# prep_data = {}
# for outfit in data.values():
#     for item in outfit:
#         for label in outfit[item]['label']:
#             if len(outfit[item]['label']['98']) > 0:
#                 prep_data[item] = outfit[item]['label']['98'][0]
#             if len(outfit[item]['label']['99']) > 0:
#                 prep_data[item] = outfit[item]['label']['99'][0]

# label_data = {}
# feats_data = {}
# for outfit in data:
#     for item in data[outfit]:
#         _, _, item_id = item.split('_')
#         feats_data[item] = loadmodel[item_id]
#         label_data[item] = prep_data[item]

# Y_train = list(label_data.values())[int(len(label_data)*0.8):]
# X_train = list(feats_data.values())[int(len(feats_data)*0.8):]

# test_index = list(label_data.values())[:int(len(label_data)*0.2)]
# test = list(feats_data.values())[:int(len(feats_data)*0.2)]

# # X_train = X_train.values.reshape(-1,28,28,1)
# # test = test.values.reshape(-1,28,28,1)

# class_names = list(labels.keys())

# X_train, X_val, Y_train, Y_val = train_test_split(X_train,
#                                                   Y_train,
#                                                   test_size=0.2,
#                                                   random_state=100)

# Y_train = np_utils.to_categorical(Y_train)
# Y_val = np_utils.to_categorical(Y_val)

# model = Sequential()
# model.add(Conv2D(32, kernel_size = (3,3), input_shape=(28,28,1), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10,activation='softmax'))

# model.compile(loss = 'categorical_crossentropy',
#               optimizer='adam',
#               metrics = ['accuracy'])

# MODEL_DIR = "./model/"

# if not os.path.exists(MODEL_DIR):
#   os.mkdir(MODEL_DIR)

# modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
# checkpointer = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss', verbose=1, save_best_only=True)

# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
#                     epochs=20, 
#                     batch_size=200, 
#                     verbose=0, 
#                     callbacks=[early_stopping_callback, checkpointer])

# print("\n Test Accuracy: %.4f" % (model.evaluate(X_val, Y_val)[1]))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

# # 오차
# y_vloss = history.history['val_loss']

# # 학습셋 오차
# y_loss = history.history['loss']

# # 그래프로 표현
# x_len = np.arange(len(y_loss))
# ax1.plot(x_len, y_vloss, marker = '.', c="red", label='Testset_loss')
# ax1.plot(x_len, y_loss, marker = '.', c='blue', label = 'Trainset_loss')

# # 그래프에 그리드를 주고 레이블을 표시
# ax1.legend(loc='upper right')
# ax1.grid()
# ax1.set(xlabel='epoch', ylabel='loss')


# # 정확도
# y_vaccuracy = history.history['val_accuracy']

# # 학습셋
# y_accuracy = history.history['accuracy']

# # 그래프로 표현
# x_len = np.arange(len(y_accuracy))
# ax2.plot(x_len, y_vaccuracy, marker = '.', c="red", label='Testset_accuracy')
# ax2.plot(x_len, y_accuracy, marker = '.', c='blue', label = 'Trainset_accuracy')

# # 그래프에 그리드를 주고 레이블을 표시
# ax2.legend(loc='lower right')
# ax2.grid()

# ax2.set(xlabel='epoch', ylabel='accuracy')

# # draw gridlines
# ax2.grid(True)
# plt.savefig('check.jpg')

# results = model.predict(test)
# results = np.argmax(results,axis = 1)
# results = pd.Series(results,name="Label")

# submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
# submission.to_csv("results_fashion_mnist.csv",index=False)
