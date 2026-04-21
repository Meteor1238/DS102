import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

def load_prep_data(folder_path, target_size=(128, 128)):
    """
    folder_path/
        ├── NORMAL/
        └── PNEUMONIA/
    """
    images = []
    labels = []
    
    classes = {'NORMAL': -1, 'PNEUMONIA': 1}
    
    for class_name, label in classes.items():
        class_dir = os.path.join(folder_path, class_name)
        file_names = os.listdir(class_dir)
        
        for file_name in tqdm(file_names, desc=class_name):
            img_path = os.path.join(class_dir, file_name)
            img = cv.imread(img_path)
            if img is not None:
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img_resized = cv.resize(img_gray, target_size, interpolation=cv.INTER_NEAREST)
                img_normalized = img_resized.astype('float32') / 255.0
                img_flattened = img_normalized.flatten()
                
                images.append(img_flattened)
                labels.append(label)
            else:
                print(f"Không thể đọc file ảnh {img_path}")
                
    X = np.array(images)
    y = np.array(labels)
    
    return X, y