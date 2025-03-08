import os
import cv2
import numpy as np
import pandas as pd
from skimage import feature, color, filters
import cv2
import numpy as np
from skimage import feature, color, filters

def calculate_texture_complexity(image, image_path):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    texture_complexity = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-6))
    return texture_complexity

def calculate_color_distribution_consistency(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    color_distribution_consistency = np.std(hist)
    return color_distribution_consistency

def calculate_object_coherence(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    object_coherence = np.sum(edges) / edges.size
    return object_coherence

def calculate_contextual_relevance(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    contextual_relevance = np.var(gradient_magnitude)
    return contextual_relevance

def calculate_image_smoothness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    image_smoothness = 1 / (1 + laplacian_var)
    return image_smoothness

def calculate_image_sharpness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    sharpness_measure = np.max(cv2.absdiff(gray_image, blurred_image))
    return sharpness_measure

def calculate_image_contrast(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_contrast = np.std(gray_image)
    return image_contrast

def process_folder_images(folder_path):
    results_df = pd.DataFrame(columns=[
        'Filename', 'TextureComplexity', 'ColorDistributionConsistency',
        'ObjectCoherence', 'ContextualRelevance', 'ImageSmoothness',
        'ImageSharpness', 'ImageContrast'
    ])



    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            TCI = calculate_texture_complexity(image, image_path)
            CDC = calculate_color_distribution_consistency(image)
            OCI = calculate_object_coherence(image)
            CR = calculate_contextual_relevance(image)
            SMO = calculate_image_smoothness(image)
            SHP = calculate_image_sharpness(image)
            CON = calculate_image_contrast(image)
            new_row = pd.DataFrame({'Filename': [filename],
            'TextureComplexity': [TCI],
            'ColorDistributionConsistency': [CDC],
            'ObjectCoherence': [OCI],
            'ContextualRelevance': [CR],
            'ImageSmoothness': [SMO],
            'ImageSharpness': [SHP],
            'ImageContrast': [CON]}) 
            results_df = pd.concat([results_df, new_row], ignore_index=True)
##################################################################################################### change csv name according to model name 
    results_df.to_csv('/path/to/your/file/indexing/COCO_Midjourney.csv',index=False)
    print(f"Processed {len(results_df)} images and saved the results")

################################################################################################################ change data path accordingly
folder_path = '/path/to/your/dataset/COCO/COCO_test/Midjourney/test/Fake/class1'
process_folder_images(folder_path)




