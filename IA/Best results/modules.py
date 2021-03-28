### Imports
from warnings import simplefilter
simplefilter('ignore')

from skimage.transform import hough_line, hough_line_peaks

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, PReLU, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as k
from matplotlib import pyplot as plt
from scipy.stats import linregress
from matplotlib import cm
from glob import glob
from tqdm import tqdm

import os, sys, shutil, random
import tensorflow as tf
import altair as alt
import pandas as pd
import numpy as np
import cv2
