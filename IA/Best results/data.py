import os, cv2, random
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)

train_data_path = 'C:/Users/Rodrigo/Google Drive/IC/Data/Train/'
train_coords_path = 'C:/Users/Rodrigo/Google Drive/IC/Data/Train/values.csv'
images_names = os.listdir(train_data_path)
images_names.remove('values.csv')
values = pd.read_csv(train_coords_path)

# Data
use_lines = True

# Gaussian
k_size = 19
std_dev = 4.0

def load_data():
    ### Preprocessing images and generating the ground truth
    x_train = [] # Training data
    gt_train = [] # Ground Truth
    x_valid = []
    y_valid = []
    random_ind = [random.randint(0, len(images_names)) for _ in range(50)]

    for i, name in tqdm(enumerate(images_names)):
        image = cv2.imread(train_data_path + name) # Loading the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converting the image to RGB
        image = image / 255.0 # Rescaling our image
        image = np.transpose(image, [2, 0, 1]) # Channels First
        
        # Getting the coords
        x1 = values[values['Name'] == name]['P1x']
        y1 = values[values['Name'] == name]['P1y']
        x2 = values[values['Name'] == name]['P2x']
        y2 = values[values['Name'] == name]['P2y']
        
        gt = np.zeros(shape=(128, 128, 3), dtype=np.float32)
        if use_lines:
            gt = cv2.line(gt, (x1, y1), (x2, y2), (255, 255, 255), 1) # Drawing the line
            gt = cv2.GaussianBlur(gt, (k_size, k_size), std_dev) # Applying Gaussian filter
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY) # Converting the ground truth to grayscale
            gt = np.reshape(gt, (128, 128, 1)) # Making it 3D
            gt = np.transpose(gt, [2, 0, 1]) # Channels First
        else:
            gt = cv2.circle(gt, (x1, y1), 1, (255, 255, 255), -1) # Drawing the first point
            gt = cv2.circle(gt, (x2, y2), 1, (255, 255, 255), -1) # Drawing the second point
            gt = cv2.GaussianBlur(gt, (k_size, k_size), std_dev) # Applying Gaussian filter
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY) # Converting the ground truth to grayscale
            gt = np.reshape(gt, (128, 128, 1)) # Making it 3D
            gt = np.transpose(gt, [2, 0, 1]) # Channels First
        
        if i in random_ind:
            x_valid.append(image)
            y_valid.append(gt)
        else:
            x_train.append(image)
            gt_train.append(gt) 

    print('Shape of x_train =', np.shape(x_train))
    print('Shape of gt_train =', np.shape(gt_train))
    print('Shape of x_valid=', np.shape(x_valid))
    print('Shape of y_valid =', np.shape(y_valid))

    x_train = tf.convert_to_tensor(x_train)
    gt_train = tf.convert_to_tensor(gt_train)
    x_valid = tf.convert_to_tensor(x_valid)
    y_valid = tf.convert_to_tensor(y_valid)

    return x_train, gt_train, x_valid, y_valid
