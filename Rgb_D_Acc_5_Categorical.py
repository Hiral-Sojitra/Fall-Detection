"""
Created on Sat Dec  3 10:07:20 2022

@author: ana_j

Model: CNN
Input layer shape : 28x28x5
Input layer shape : 28x28x4

Depth : 5 where (RGB (3) + Depth (1) + Acc(1))

Depth : 4 where (RGB (3) + Depth (1))


"""

# ========================= Libraries =========================================
import cv2
import os
import sys
import glob
import numpy as np
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import seaborn as sns
from PIL import Image
import keras

from keras.layers import BatchNormalization
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from keras.applications import VGG16

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import argparse
import locale
import os
import io
import imageio
import open3d as o3d


from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import tensorflow as tf

np.random.seed(100)
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils import to_categorical
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# ================= Folders ===================================================

fall_folder = 'URFD_images_not_segmented/Falls/'

# Label files
matrix_csv = 'urfall-matrix-acc.csv'
# Label files with avg acc
# matrix_csv = 'urfall-matrix-avg-acc.csv'

# ========================= Constants =========================================

width_shape, height_shape = 28, 28 # shape of new images 

# ========================= Files Reading  ====================================

df_matrix= pd.read_csv(matrix_csv, index_col=0)
print(df_matrix.head())

# ===================== Functions =============================================

# labels 0/1
def label_img(img_name):
    split = img_name.split("-")
    # print(split) # ['fall', '01', 'cam0', 'd', '001.png']
    fall_name = split[0] + "-" + split[1]
    # print(fall_name) # fall-01
    frame = int(split[-1].split(".")[0])
    # print(frame) # 1
    # print(type(frame))
    row_fall = df_matrix.loc[fall_name]
    # print(row_fall) # fall-01        1     -1           3.16670  ...  1899.5366  1055.9988  0.047310
    frame_label = row_fall.loc[row_fall["frame"] == frame]    
    # print(frame_label) # fall-01        1     -1            3.1667  ...  1899.5366  1055.9988  0.04731
    label_int = int (frame_label.label)
    # sv_total = frame_label.SV_total_sync
    # print(label_int) # -1
    return label_int 

# frames
def frame_img(img_name):
    split = img_name.split("-")
    # print(split) # ['fall', '01', 'cam0', 'd', '001.png']
    fall_name = split[0] + "-" + split[1]
    # print(fall_name) # fall-01
    frame = int(split[-1].split(".")[0])
    # print(frame) # 1
    # print(type(frame))
    row_fall = df_matrix.loc[fall_name]
    # print(row_fall) # fall-01        1     -1           3.16670  ...  1899.5366  1055.9988  0.047310
    frame_label = row_fall.loc[row_fall["frame"] == frame]    
    # print(frame_label) # fall-01        1     -1            3.1667  ...  1899.5366  1055.9988  0.04731
    frame_int = int (frame_label.frame)
    # sv_total = frame_label.SV_total_sync
    # print(label_int) # -1
    return frame_int 


# accelerometer data    
def acc_data_img(img_name):
    split = img_name.split("-")
    # print(split) # ['fall', '01', 'cam0', 'd', '001.png']
    fall_name = split[0] + "-" + split[1]
    # print(fall_name) # fall-01
    frame = int(split[-1].split(".")[0])
    # print(frame) # 1
    # print(type(frame))
    row_fall = df_matrix.loc[fall_name]
    # print(row_fall) # fall-01        1     -1           3.16670  ...  1899.5366  1055.9988  0.047310
    frame_label = row_fall.loc[row_fall["frame"] == frame]
    # print(frame_label) # fall-01        1     -1            3.1667  ...  1899.5366  1055.9988  0.04731
    acc_data = float(frame_label.SV_total_sync)  
#     print(acc_data) # -1
    return acc_data


def load_img_lbl_acc_frame():    
    images_d = []
    images_rgb = []
    labels_d = []
    labels_rgb = []
    acc_d = []
    acc_rgb = []
    frames = []
    
    for img_folder in tqdm(os.listdir(fall_folder)):
        if img_folder != '.ipynb_checkpoints':
#             print(img_folder) # fall-01-cam0-d
            split = img_folder.split("-")[3]
            full_path = os.path.join(fall_folder,img_folder)
            # print(full_path) # URFD_images_not_segmented/Falls/fall-01-cam0-d
            for img_file in os.listdir(full_path): 
                if img_file.endswith('png'):
                    if split == 'd':
#                         print(img_file) # fall-01-cam0-d-001.png                        
                        img = cv2.imread(os.path.join(full_path, img_file),0)
                        # img = imageio.imread(os.path.join(full_path, img_file))
                        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        img = resize(img,(height_shape,width_shape))
                        img = np.expand_dims(img,axis=-1)
                        images_d.append(img)
                        labels_d.append([label_img(img_file)])
                        acc_d.append([acc_data_img(img_file)])
                        frames.append([frame_img(img_file)])                         
                    elif split == 'rgb':
#                         print(os.path.join(full_path, img_file))
                        img = cv2.imread(os.path.join(full_path, img_file))
                        # img = imageio.imread(os.path.join(full_path, img_file))
#                         print(img_file)
                        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
                        img = resize(img,(height_shape,width_shape))
                        images_rgb.append(img)                        
                        labels_rgb.append([label_img(img_file)]) 
                        acc_rgb.append([acc_data_img(img_file)])
                        frames.append([frame_img(img_file)])                         
                else:
                    print("error")
                    
    # return images_d, images_rgb, acc_data, frames, labels 
    return images_d, images_rgb, labels_d, acc_d, frames


# =============================================================================

images_d, images_rgb, labels_d, acc_d, frames = load_img_lbl_acc_frame() 

#========================= Image visualization  ===============================
""" 
plt.imshow(images_d[83])  
print((images_d[83]).shape) 

plt.imshow(images_r[83])  # error while printing imgs
print((images_r[83]).shape) 
"""
# ====================  Converting into np. array =============================

# Images 
x_images_d = np.array(images_d) 
print('X_images_d:',x_images_d.shape) 

x_images_rgb = np.array(images_rgb)
print('X_images_rgb:',x_images_rgb.shape) 

tensor_rgbd = tf.concat((x_images_d, x_images_rgb), axis=3)
print('tensor_rgbd:',tensor_rgbd.shape)
tensor_rgbd = tensor_rgbd.numpy()
print(type(tensor_rgbd)) 


# Accelerometer data 
x_acc = np.array(acc_d)
print('X_acc:',x_acc.shape)

# print(X_acc[:,np.newaxis,np.newaxis,:].shape)


# Frames
lbl_frames = np.array(frames)
print('lbl_frames:',lbl_frames.shape)

# Labels
y = np.array(labels_d)
print('y:',y.shape)


# indexes
# indexes = np.arange(X_images_d.shape[0])
# print(indexes)


# ====================  train/test split with test_size=0.2  ========================================

X_train_img, X_test_img, X_train_acc, X_test_acc, y_train_lbls, y_test_lbls = train_test_split(tensor_rgbd, x_acc, to_categorical(y,num_classes = 2), test_size = 0.20, random_state = 0)

# ====== Uncomment this if you want to include only depth imgs

# X_train_img, X_test_img, X_train_acc, X_test_acc, y_train_lbls, y_test_lbls = train_test_split(x_images_d, x_acc, to_categorical(y,num_classes = 2), test_size = 0.20, random_state = 0)


# ====================  normalization of accelerometer data  ==================
scaler = MinMaxScaler()
X_train_acc = scaler.fit_transform(X_train_acc)
X_test_acc = scaler.transform(X_test_acc)


# ====================  

x_acc_exp_train = np.zeros((4792,28,28))
print((x_acc_exp_train.shape))

for i in range(0,4792):
    x_acc_exp_train[i,:,:] = X_train_acc[i,0] * np.ones((28,28))
        
print(x_acc_exp_train.shape)

x_acc_exp_train = x_acc_exp_train.reshape(4792,28,28,1)
print(x_acc_exp_train.shape)


tensor_rgbd_acc_train = tf.concat((X_train_img, x_acc_exp_train), axis=3)
tensor_rgbd_acc_train = tensor_rgbd_acc_train.numpy()
print('tensor_rgbd_acc_train:',tensor_rgbd_acc_train.shape)



x_acc_exp_test = np.zeros((1198,28,28))
print((x_acc_exp_test .shape))

for i in range(0,1198):
    x_acc_exp_test[i,:,:] = X_test_acc[i,0] * np.ones((28,28))
        
print(x_acc_exp_train.shape)

x_acc_exp_test = x_acc_exp_test.reshape(1198,28,28,1)
print(x_acc_exp_test.shape)


tensor_rgbd_acc_test = tf.concat((X_test_img, x_acc_exp_test), axis=3)
tensor_rgbd_acc_test = tensor_rgbd_acc_test.numpy()
print('tensor_rgbd_acc_test:',tensor_rgbd_acc_test.shape)



# Train
print('tensor_rgbd_acc_train: ', tensor_rgbd_acc_train.shape)
print('y_train_lbls : ', y_train_lbls.shape)


# Test
print('tensor_rgbd_acc_test: ' , tensor_rgbd_acc_test.shape)
print('y_test_lbls :' , y_test_lbls.shape)



#============================= CNN ===========================================

depth = 5 # change this to 4 if you want to inlude rgb and depth
INPUT_SHAPE = (width_shape, height_shape, depth)
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)
conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

flat = keras.layers.Flatten()(drop2)  #Flatten the matrix to get it ready for dense.

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)
hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

out = keras.layers.Dense(2, activation='sigmoid')(drop4)   #units=1 gives error

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='categorical_crossentropy',   #Try also binary_crossentropy 
                metrics=['accuracy'])
print(model.summary())



# ======================== Fit the model rgb + depth + acc  =====================================

history = model.fit(tensor_rgbd_acc_train, 
                    y_train_lbls,
                    batch_size=64,
                    verbose=1,
                    epochs=50,
                    validation_split=0.1,
                    shuffle=False
                    )

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(tensor_rgbd_acc_test), np.array(y_test_lbls))[1]*100))

# ======================== Fit the model  rgb + depth  =====================================
# history = model.fit(X_train_img,
#                     y_train_lbls,
#                     batch_size=64,
#                     verbose=1,
#                     epochs=50,
#                     validation_split=0.1,
#                     shuffle=False
#                     )

# print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test_img), np.array(y_test_lbls))[1]*100))


# ====================== Evaluating Model Performance =========================

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance - RGB+depth+Acc', fontsize=12) # Change this label to whatever beter describes the plot
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


# Get predictions to calculate precision, recall, F1 score
y_pred = model.predict(tensor_rgbd_acc_test)

np.save("y_pred_rgb_d_acc_5_categorical", y_pred)
np.save("y_test_rgb_d_acc_5_categorical", y_test_lbls)

# Calculate precision, recall, F1 score. In the table, false - not a fall; true - a fall
from sklearn.metrics import classification_report
print(classification_report(y_test_lbls, y_pred > 0.5))


# Create confusion matrix

y_test_matrix = []
for i in y_test_lbls:
    y_test_matrix.append(i[0])
    
y_pred_matrix = []
y_pred_bool = y_pred > 0.5
for i in y_pred_bool:
    y_pred_matrix.append(i[0])

import matplotlib.pyplot as plt

import seaborn as sn
data = {'y_Actual':   y_test_matrix,
        'y_Predicted': y_pred_matrix
        }
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])


sn.heatmap(confusion_matrix,xticklabels=['Fall', 'Not Fall'], yticklabels=[ 'Fall', 'Not Fall'],fmt='g', annot=True)
plt.show()



# ======= auc plot

fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_test_lbls, y_pred.argmax(axis=1))
auc_cnn = auc(fpr_cnn, tpr_cnn)


import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_cnn, tpr_cnn, label='CNN (area = {:.3f})'.format(auc_cnn))


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# ====================== Save the model  ======================================

model.save('UR_RgbD_Acc_5.h5')

y_pred_cnn_categorical = np.load("y_pred_rgb_d_acc_5_categorical.npy")
y_test_cnn_no_acc = np.load("y_test_rgb_d_acc_5_categorical.npy")

model_rgbd_acc_5_categorical = keras.models.load_model('UR_RgbD_Acc_5.h5')


# fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_test_cnn_no_acc, y_pred_cnn_no_acc)
# auc_cnn = auc(fpr_cnn, tpr_cnn)






# ====================== Load the model  ======================================

#Load a model 
modelLoad = keras.models.load_model("UR_RgbD_Acc_5.h5")



modelLoad.fit(tensor_rgbd_acc_train,y_train_lbls)
print("Test_Accuracy: {:.2f}%".format(modelLoad.evaluate(np.array(tensor_rgbd_acc_test), np.array(y_test_lbls))[1]*100))
y_pred = modelLoad.predict(tensor_rgbd_acc_test)

np.save("y_pred", y_pred)
np.save("y_test", y_test_lbls)

# Calculate precision, recall, F1 score. In the table, false - not a fall; true - a fall
from sklearn.metrics import classification_report
print(classification_report(y_test_lbls, y_pred > 0.5))


# Create confusion matrix

y_test_matrix = []
for i in y_test_lbls:
    y_test_matrix.append(i[0])
    
y_pred_matrix = []
y_pred_bool = y_pred > 0.5
for i in y_pred_bool:
    y_pred_matrix.append(i[0])

import matplotlib.pyplot as plt

import seaborn as sn
data = {'y_Actual':   y_test_matrix,
        'y_Predicted': y_pred_matrix
        }
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])


sn.heatmap(confusion_matrix,xticklabels=['Fall', 'Not Fall'], yticklabels=[ 'Fall', 'Not Fall'],fmt='g', annot=True)
plt.show()

