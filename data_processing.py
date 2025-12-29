import pandas as pd
import numpy as np
import tensorflow as tf
import pydicom
from typing import Tuple

#fix known file path issue in dataset
#original paths had wrong suffix, correct to actual dicom naming
def correct_file_path(path: str) -> str:
    return path.replace('000000.dcm', '1-1.dcm')

#load metadata csvs and encode pathology labels
#benign/benign_without_callback -> 0, malignant -> 1
def load_metadata(train_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    #encode labels for both sets
    train_df['pathology'] = train_df['pathology'].map({
        'BENIGN': 0,
        'BENIGN_WITHOUT_CALLBACK': 0,
        'MALIGNANT': 1
    })
    
    test_df['pathology'] = test_df['pathology'].map({
        'BENIGN': 0,
        'BENIGN_WITHOUT_CALLBACK': 0,
        'MALIGNANT': 1
    })
    
    #fix file paths
    train_df['image file path'] = train_df['image file path'].apply(correct_file_path)
    test_df['image file path'] = test_df['image file path'].apply(correct_file_path)
    
    return train_df, test_df

#load single dicom image and preprocess for resnet
#converts grayscale to rgb, normalizes, resizes with padding
def load_dicom_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    try:
        dicom = pydicom.dcmread(image_path)
        img = dicom.pixel_array
        
        #cast to float and add channel dim
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, -1)
        
        #convert grayscale to rgb for resnet compatibility
        img = tf.image.grayscale_to_rgb(img)
        
        #resize with padding to maintain aspect ratio
        img = tf.image.resize_with_pad(img, target_size[0], target_size[1])
        
        #normalize to 0-1 range
        img = img / tf.reduce_max(img)
        
        return img
    except Exception as e:
        print(f"error loading {image_path}: {e}")
        return None

#batch load all images from dataframe
#returns numpy array of preprocessed images
def load_images_batch(df: pd.DataFrame, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    images = []
    failed = 0
    
    for idx, path in enumerate(df['image file path']):
        img = load_dicom_image(path, target_size)
        
        if img is not None:
            images.append(img.numpy())
        else:
            failed += 1
            #append black image as placeholder for failed loads
            images.append(np.zeros((target_size[0], target_size[1], 3)))
    
    if failed > 0:
        print(f"warning: {failed}/{len(df)} images failed to load")
    
    return np.array(images)

#prepare train and test data from metadata
#returns images and labels as numpy arrays
def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                target_size: Tuple[int, int] = (224, 224)) -> Tuple:
    
    print(f"loading {len(train_df)} training images...")
    train_images = load_images_batch(train_df, target_size)
    train_labels = np.array(train_df['pathology'])
    
    print(f"loading {len(test_df)} test images...")
    test_images = load_images_batch(test_df, target_size)
    test_labels = np.array(test_df['pathology'])
    
    print(f"train set: {train_images.shape}, test set: {test_images.shape}")
    
    return train_images, train_labels, test_images, test_labels
