from PIL import ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy

# Loss Function Setup Dice loss, Combination Loss
smooth = 1.

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# Load Trained Model
model = load_model('UNetMobileV2_FullData_CustomLoss_TF_NoFreeze_DataAugmentation', compile = True, custom_objects={"dsc": dsc, "bce_dice_loss": bce_dice_loss})
model.summary()

# Plotting Function
def display(display_list):
    plt.figure(figsize=(15,15))
    
    title = ['Input Image', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

# Auto Segmentation with saved model on chosen CT image slice
# Return predicted tumour and its tumour centroid  
def prediction(img, saved_model=model):
    # create array of the right shape (256, 256)
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    image = img

    # Resize to 256 by 256
    size = (256, 256)
    image = ImageOps.fit(image, size)

    # Normalise pixel values to 0 - 1 range
    image_arr = np.asarray(image)
    pixels = image_arr.astype('float32')
    pixels /= 255.0


    data[0] = pixels
    pred_mask = saved_model.predict(data)
    

    # Remove white noise by setting threshold value 
    threshold_percentage = 0.5
    pred_mask = np.where(pred_mask > threshold_percentage, 1, 0)
    display([data[0], pred_mask[0]])
    plt.savefig('predicted_mask.png')

    pred_mask_com = ndimage.measurements.center_of_mass(pred_mask)
    print('Predicted Tumour Centroid: ', pred_mask_com)

    return pred_mask, pred_mask_com



