import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import cv2
import tensorflow as tf
from keras.models import model_from_json

# Load the U-Net model
def load_unet_model(model_path="Segmentation_Unet.hdf5", json_path="segmentation_model_best.json"):
    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    loaded_model.load_weights(model_path)
    return loaded_model

# Example usage:
loaded_model = load_unet_model()


def load_preprocess_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image / 256.0
    return image

# predict_segmentation_mask function to accept the loaded model
def predict_segmentation_mask(image_path, model):
    """reads a brain MRI image
    Returns the segmentation mask of the image
    """
    img = Image.open(image_path)
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    img -= img.mean()
    img /= img.std()
    img = np.reshape(img, (1, 256, 256, 3))  # Reshape to match the model input shape
    predict = model.predict(img)

    return predict.reshape(256, 256)  # Remove the unnecessary third dimension



def plot_MRI_predicted_mask(original_img, predicted_mask):
    """
    Inputs: image and mask
    Outputs: plot both image and mask side by side
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
    axes[0].imshow(original_img)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_title('Original MRI')

    axes[1].imshow(predicted_mask, cmap='gray')  # cmap='gray' for binary masks
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_title('Predicted Mask')

    fig.tight_layout()
    filename = 'pair' + str(random.randint(100, 1000)) + str(random.randint(100, 1000)) + '.png'
    plt.savefig(filename)

    print('File saved successfully')
    print(filename)

    return filename



def final_fun_1(image_path):
    '''  Input: Image path through the path upload method
        Returns : combined image of original and predicted mask
    '''
    # Load the U-Net model
    loaded_model = load_unet_model()

    # Preprocessing the inputs
    image = load_preprocess_image(image_path)
    mask = predict_segmentation_mask(image_path, loaded_model)
    
    combined_img = plot_MRI_predicted_mask(original_img=image, predicted_mask=mask) 
    return combined_img
