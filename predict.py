import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as pil_image
import cv2
import glob
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
import scipy
from scipy import ndimage
from skimage.measure import label, regionprops