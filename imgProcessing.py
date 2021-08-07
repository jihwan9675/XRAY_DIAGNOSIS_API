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


K.set_learning_phase(False)

def load_model(json_path, h5_path):
    with open(json_path, "r") as f:
        loaded_model_json = f.read()
    
    tl_model = model_from_json(loaded_model_json)
    tl_model.load_weights(h5_path)
    
    return tl_model


def preprocess_input(img_path):
    # img = pil_image.open(img_path).resize((224, 224))
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, dsize=(600,600))
    cla = clahe.apply(img)
    img = cv2.cvtColor(cla, cv2.COLOR_GRAY2RGB)
    img_arr = np.asarray(img)[:, :, :3] / 255.
    img_tensor = np.expand_dims(img_arr, 0)
    print(img_tensor.shape)
    return img_arr, img_tensor

def generate_cam(model, img_path, class_idx):
    
    ## img_path -> preprocessed image tensor
    img_arr, img_tensor = preprocess_input(img_path)
    
    ## preprocessed image tensor -> last_conv_output, predictions
    get_output = K.function([model.layers[0].input], [model.layers[-4].output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img_tensor])
    
    conv_outputs = conv_outputs[0, :, :, :]
    class_weights = model.layers[-1].get_weights()[0]
    
    ## generate cam
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, class_idx]):
        cam += w * conv_outputs[:, :, i]
        
    cam /= np.max(cam)
    cam = cv2.resize(cam, (600, 600))
    
    return img_arr, cam, predictions

def generate_grad_cam(model, img_path, class_idx):
    img_arr, img_tensor = preprocess_input(img_path)
    y_c = model.layers[-1].output.op.inputs[0][0, class_idx]
    layer_output = model.layers[-4].output
    
    grads = K.gradients(y_c, layer_output)[0]
    gradient_fn = K.function([model.input], [layer_output, grads, model.layers[-1].output])
    
    conv_output, grad_val, predictions = gradient_fn([img_tensor])
    conv_output, grad_val = conv_output[0], grad_val[0]
    
    weights = np.mean(grad_val, axis=(0, 1))
    cam = np.dot(conv_output, weights)
    
    cam = cv2.resize(cam, (600, 600))
    
    ## Relu
    cam = np.maximum(cam, 0)
    
    cam = cam / cam.max()
    
    return img_arr, cam, predictions

def generate_bbox(img, cam, threshold):
    labeled, nr_objects = ndimage.label(cam > threshold)
    props = regionprops(labeled)
    return props


def dcmtojpg(path):
    ds = pydicom.read_file(path)
    img = ds.pixel_array
    print(img)
    cv2.imwrite('RY.jpeg', img)
    #scipy.misc.imsave('RY_'+path, img)
    return img

#dcmtojpg("aa.dcm")

img_path = '1.jpeg'
JSON_PATH = "resnet_CAM_new.json"
H5_PATH = "resnet_CAM_new.h5"
## 1. load model
model = load_model(JSON_PATH, H5_PATH)
class_indices = ['NORMAL','PNEUMONIA']
# img, cam, predictions = generate_cam(model, img_path, class_idx)
img, cam, predictions = generate_cam(model, img_path, 1)
pred_values = np.squeeze(predictions, 0)
top1 = np.argmax(pred_values)
top1_value = round(float(pred_values[top1]), 3)
props = generate_bbox(img, cam, 0.3) ########################################## I have to modify this threshold.


plt.imshow(cam, cmap='jet')
ax = plt.gca()


heat_img = np.zeros((600,600,3), np.uint8)
bound_img = np.zeros((600,600,3), np.uint8)

for b in props:
    bbox = b.bbox
    xs = bbox[1]
    ys = bbox[0]
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    # heat_img[bbox[1]:bbox[0], bbox[3]:bbox[2]] = cam[bbox[1]:bbox[0], bbox[3]:bbox[2]]

    rect = patches.Rectangle((xs, ys), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
cv2.imwrite('img.jpeg', img*255)
cv2.imwrite('cam.jpeg', cam*255)
cv2.imwrite('heatcam.jpeg', heat_img)
# print(class_indices[top1])
# print(top1_value)
# plt.savefig('result.jpeg')

# fig, axes = plt.subplots(4, 2, figsize=(8, 4))

# for i, s in enumerate(samples):
#     img_set = s['target']
#     img_path = s['img_path']
#     class_idx = s['class_idx']
#     img, cam, predictions = generate_cam(model, img_path, class_idx)
#     pred_values = np.squeeze(predictions, 0)
#     top1 = np.argmax(pred_values)
#     top1_value = round(float(pred_values[top1]), 3)
#     props = generate_bbox(img, cam, 0.3) ########################################## I have to modify this threshold.
    
#     axes[0, i].imshow(img)
#     axes[1, i].imshow(cam)
#     axes[2, i].imshow(img)
#     axes[2, i].imshow(cam, cmap='jet', alpha=0.5)
    
#     axes[3, i].imshow(img)
#     for b in props:
#         bbox = b.bbox
#         xs = bbox[1]
#         ys = bbox[0]
#         w = bbox[3] - bbox[1]
#         h = bbox[2] - bbox[0]

#         rect = patches.Rectangle((xs, ys), w, h, linewidth=2, edgecolor='r', facecolor='none')
#         axes[3, i].add_patch(rect)
    
#     axes[0,i].axis('off')
#     axes[1,i].axis('off')
#     axes[2,i].axis('off')
#     axes[3,i].axis('off')
    
#     axes[0, i].set_title("pred: {} - {}".format(class_indices[top1], top1_value), fontsize=15)
    
