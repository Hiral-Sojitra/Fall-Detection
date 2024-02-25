# Libraries
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


#================= Folders ====================================================

fall_folder = 'URFD_images_not_segmented/Falls/'

# Label files
matrix_csv = 'urfall-matrix.csv'

#========================= Constants ==========================================

width_shape, height_shape = 28, 28 # shape of new images (resize is applied)

#========================= Files Reading  =====================================

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


# ==============================================================================================

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
X_images_d = np.array(images_d) 
print('X_images_d:',X_images_d.shape)

# X_images_d_s = np.swapaxes(X_images_d, 3, 1)
# print(X_images_d_s.shape)

# stacked_img = np.stack((X_images_d,)*3, axis=-1)
# print('stacked_img:',stacked_img.shape)

X_images_rgb = np.array(images_rgb)
print('X_images_rgb:',X_images_rgb.shape)

# X_images_rgb_s = np.swapaxes(X_images_rgb, 3, 1)
# print(X_images_rgb_s.shape)


tensor_rgbd = tf.concat((X_images_d, X_images_rgb), axis=3)
print('tensor_rgbd:',tensor_rgbd.shape)
tensor_rgbd = tensor_rgbd.numpy()
print(type(tensor_rgbd))

# X_stack2 = np.concatenate((stacked_img, X_images_rgb),axis=0)
# # stackx =np.concatenate((X_images_d, X_images_rgb))
# print('X_stack2:',X_stack2.shape)

# Accelerometer data 
X_acc = np.array(acc_d)
print('X_acc:',X_acc.shape)


# X_acc_4 = np.expand_dims(X_acc, 1).shape
# # X_acc_4 = np.expand_dims(X_acc, 2).shape
# print('X_acc_4:',X_acc_4)

# tensor_rgbd_acc = tf.concat((tensor_rgbd, X_acc_4), axis=1)
# print('tensor_rgbd:',tensor_rgbd.shape)


# stacked_img_acc = np.stack((X_images_d,)*3, axis=-1)
# print('stacked_img:',stacked_img.shape)


# Frames
lbl_frames = np.array(frames)
print('lbl_frames:',lbl_frames.shape)

# Labels
y = np.array(labels_d)
print('y:',y.shape)


#indexes
indexes = np.arange(X_images_d.shape[0])
print(indexes)


# X_stack2 = np.concatenate((X_images, lbl_frames),axis=1)
# print('X_stack2:',X_stack2.shape)


# ====================  train/test split with test_size=0.2  ========================================

X_train_img, X_test_img, X_train_acc, X_test_acc, y_train_lbls, y_test_lbls = train_test_split(tensor_rgbd, X_acc, to_categorical(y,num_classes = 2), test_size = 0.20, random_state = 0)

# Train
print('X_train_img : ', X_train_img.shape)

print('X_train_acc : ', X_train_acc.shape)

print('y_train_lbls : ', y_train_lbls.shape)


# Test
print('X_test_img : ' , X_test_img.shape)

print('X_test_acc : ', X_test_acc.shape)

print('y_test_lbls :' , y_test_lbls.shape)


# =========================  Normalizing the Data =============================

scaler = MinMaxScaler()
X_train_acc = scaler.fit_transform(X_train_acc)
X_test_acc = scaler.transform(X_test_acc)

#============================= CNN ====================================
INPUT_SHAPE = (width_shape, height_shape, 4)
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
                loss='categorical_crossentropy',   #Check between binary_crossentropy and categorical_crossentropy
                metrics=['accuracy'])
print(model.summary())
    # return model



# ======================== Fit the model  =====================================

history = model.fit(X_train_img,
                    y_train_lbls,
                    batch_size=64,
                    verbose=1,
                    epochs=50,
                    validation_split=0.1,
                    shuffle=False
                    )

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test_img), np.array(y_test_lbls))[1]*100))

# ====================== Evaluating Model Performance =========================


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
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
y_pred = model.predict(X_test_img)

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


# ====================== Save the model  ======================================

#Save the model
model.save('RGB_Depth_4.h5') 



