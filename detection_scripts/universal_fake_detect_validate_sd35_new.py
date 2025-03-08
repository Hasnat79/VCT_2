import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/UniversalFakeDetect")
import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
import random
import shutil
import subprocess
import tempfile
import zipfile
import re
import requests
from scipy.ndimage.filters import gaussian_filter

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)


def calculate_metrics(y_true, y_pred, thres):
    # Convert predictions to binary based on threshold
    y_pred_binary = y_pred > thres
    
    # Calculate accuracy for real images, fake images, and overall
    r_acc = accuracy_score(y_true[y_true==0], y_pred_binary[y_true==0])
    f_acc = accuracy_score(y_true[y_true==1], y_pred_binary[y_true==1])
    acc = accuracy_score(y_true, y_pred_binary)
    
    # Calculate precision and recall
    # For fake detection: positive class is 1 (fake), negative class is 0 (real)
    
    # True positives: fake images correctly classified as fake
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    # False positives: real images incorrectly classified as fake
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    # False negatives: fake images incorrectly classified as real
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    # True negatives: real images correctly classified as real
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    
    # Calculate precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return r_acc, f_acc, acc, precision, recall, f1


def validate(model, loader, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader)))
        for img, label in tqdm(loader):
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Metrics based on 0.5 threshold
    r_acc0, f_acc0, acc0, precision0, recall0, f1_0 = calculate_metrics(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0, precision0, recall0, f1_0

    # Metrics based on the best threshold
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1, precision1, recall1, f1_1 = calculate_metrics(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, precision0, recall0, f1_0, r_acc1, f_acc1, acc1, precision1, recall1, f1_1, best_thres


def recursively_read(rootdir, must_contain="", exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    """
    Recursively read image files from a directory
    
    Args:
        rootdir: Root directory to search
        must_contain: Optional string that must be in the path
        exts: List of valid file extensions
        
    Returns:
        List of image file paths
    """
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if "." in file and file.split('.')[1].lower() in exts and must_contain in os.path.join(r, file):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    """
    Get a list of image file paths from a directory or pickle file
    
    Args:
        path: Path to directory or pickle file
        must_contain: Optional string that must be in the path
        
    Returns:
        List of image file paths
    """
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class LocalImageDataset(Dataset):
    def __init__(self, real_folder, fake_folder, max_sample, arch, 
                 jpeg_quality=None, gaussian_sigma=None):
        """
        A dataset that uses local folders for both real and fake images
        
        Args:
            real_folder: Path to the folder containing real images
            fake_folder: Path to the folder containing fake images (e.g., SD 3.5 generated)
            max_sample: Maximum number of samples to use from each category
            arch: Architecture name for normalization
            jpeg_quality: Optional JPEG quality for augmentation
            gaussian_sigma: Optional Gaussian blur sigma for augmentation
        """
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # Load real images from local folder
        print(f"Loading real images from local folder: {real_folder}")
        self.real_list = get_list(real_folder)
        
        if max_sample is not None and max_sample < len(self.real_list):
            random.shuffle(self.real_list)
            self.real_list = self.real_list[:max_sample]
            
        print(f"Using {len(self.real_list)} real images from {real_folder}")
        
        # Load fake images from local folder
        print(f"Loading fake images from local folder: {fake_folder}")
        self.fake_list = get_list(fake_folder)
        
        if max_sample is not None and max_sample < len(self.fake_list):
            random.shuffle(self.fake_list)
            self.fake_list = self.fake_list[:max_sample]
            
        print(f"Using {len(self.fake_list)} fake images from {fake_folder}")
        
        # Ensure balanced dataset
        self.sample_count = min(len(self.real_list), len(self.fake_list))
        self.real_list = self.real_list[:self.sample_count]
        self.fake_list = self.fake_list[:self.sample_count]
        
        print(f"Final dataset contains {self.sample_count} images of each type (total: {self.sample_count*2})")
        
        # Setup transformations
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])
        
    def __len__(self):
        # Total samples = real + fake
        return self.sample_count * 2
    
    def __getitem__(self, idx):
        is_fake = idx >= self.sample_count
        
        if is_fake:
            # Get fake image from fake folder
            img_path = self.fake_list[idx - self.sample_count]
            label = 1
        else:
            # Get real image from real folder
            img_path = self.real_list[idx]
            label = 0
            
        # Open and convert image
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Apply augmentations if specified
            if self.gaussian_sigma is not None:
                img = gaussian_blur(img, self.gaussian_sigma) 
            if self.jpeg_quality is not None:
                img = png2jpg(img, self.jpeg_quality)
                
            # Apply transformations
            img = self.transform(img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a black image as fallback
            img = torch.zeros(3, 224, 224)
            
        return img, label


def download_from_gdrive(url, output_folder):
    """
    Download files from Google Drive and extract if it's a zip file
    
    Args:
        url: Google Drive URL
        output_folder: Local folder to save downloaded files
        
    Returns:
        Path to the folder containing the images
    """
    print(f"Processing Google Drive URL: {url}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract folder ID if it's a folder link
    folder_id = None
    file_id = None
    
    if 'drive.google.com' in url:
        if '/file/d/' in url:
            # Format: https://drive.google.com/file/d/FILE_ID/view...
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            # Format: https://drive.google.com/open?id=FILE_ID
            file_id = url.split('id=')[1].split('&')[0]
        elif '/folders/' in url:
            # Format: https://drive.google.com/drive/folders/FOLDER_ID
            folder_id = url.split('/folders/')[1].split('?')[0].split('/')[0]
    
    if not file_id and not folder_id:
        if os.path.exists(url):
            print(f"Using {url} as a local path")
            return url
        else:
            print("Could not extract Google Drive ID from the URL and it's not a valid local path.")
            return None
    
    if folder_id:
        return download_gdrive_folder(folder_id, output_folder)
    else:
        return download_gdrive_file(file_id, output_folder)


def download_gdrive_file(file_id, output_folder):
    """
    Download a single file from Google Drive
    """
    print(f"Downloading Google Drive file with ID: {file_id}")
    
    # Download path
    temp_file = os.path.join(output_folder, "temp_download")
    
    try:
        # Try using gdown if available
        try:
            import gdown
            print("Using gdown to download the file...")
            output = gdown.download(id=file_id, output=temp_file, quiet=False)
            if not output:
                raise Exception("gdown download failed")
        except (ImportError, Exception) as e:
            print(f"gdown error: {e}")
            print("Falling back to direct download method...")
            
            # Direct download using requests
            print("Downloading using requests...")
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            session = requests.Session()
            
            response = session.get(url, stream=True)
            token = None
            
            # Check if we hit the download warning page
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
                    
            if token:
                url = f"{url}&confirm={token}"
                response = session.get(url, stream=True)
                
            # Download the file
            with open(temp_file, 'wb') as f:
                total_length = response.headers.get('content-length')
                if total_length:
                    total_length = int(total_length)
                    with tqdm(total=total_length, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=4096):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    f.write(response.content)
        
        # Check if it's a zip file and extract
        if zipfile.is_zipfile(temp_file):
            print("Extracting zip file...")
            extract_dir = os.path.join(output_folder, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"Files extracted to {extract_dir}")
            # Clean up the zip file
            os.remove(temp_file)
            return extract_dir
        else:
            # If it's a single image, move it to the images folder
            images_dir = os.path.join(output_folder, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Try to detect the file type
            with open(temp_file, 'rb') as f:
                header = f.read(20)
                file_extension = None
                
                # Simple file signature checking
                if header.startswith(b'\xff\xd8'):  # JPEG
                    file_extension = 'jpg'
                elif header.startswith(b'\x89PNG'):  # PNG
                    file_extension = 'png'
                elif header.startswith(b'GIF8'):  # GIF
                    file_extension = 'gif'
                else:
                    file_extension = 'unknown'
            
            if file_extension != 'unknown':
                shutil.move(temp_file, os.path.join(images_dir, f"image.{file_extension}"))
            else:
                print("Warning: Downloaded file doesn't appear to be an image or zip.")
                shutil.move(temp_file, os.path.join(images_dir, "unknown_file"))
                
            return images_dir
            
    except Exception as e:
        print(f"Error downloading file from Google Drive: {e}")
        return None


def download_gdrive_folder(folder_id, output_folder):
    """
    Download all files from a Google Drive folder
    """
    print(f"Downloading Google Drive folder with ID: {folder_id}")
    download_folder = os.path.join(output_folder, f"folder_{folder_id}")
    os.makedirs(download_folder, exist_ok=True)
    
    # Method 1: Try using gdown for folder download
    try:
        import gdown
        print("Attempting to download folder using gdown...")
        output = gdown.download_folder(id=folder_id, output=download_folder, quiet=False, use_cookies=False)
        if output:
            print(f"Successfully downloaded folder to {download_folder}")
            return download_folder
    except (ImportError, Exception) as e:
        print(f"gdown folder download failed: {e}")
    
    # Method 2: Try using direct API to list files in the folder and download them individually
    try:
        print("Attempting to list files in folder...")
        # Use requests to get folder contents (may not work due to Google restrictions)
        session = requests.Session()
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        response = session.get(folder_url)
        content = response.text
        
        # Extract file IDs using regex (this is a simple approach and might break)
        file_ids = []
        # Look for patterns like "https://drive.google.com/file/d/FILE_ID/view"
        pattern = r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"
        matches = re.findall(pattern, content)
        file_ids.extend(matches)
        
        if not file_ids:
            print("Could not find file IDs in the folder.")
            # Additional logic to try using other methods could be added here
        else:
            print(f"Found {len(file_ids)} files in the folder.")
            success_count = 0
            
            for idx, file_id in enumerate(file_ids):
                print(f"Downloading file {idx+1}/{len(file_ids)} with ID: {file_id}")
                file_download_path = os.path.join(download_folder, f"file_{idx}")
                try:
                    # Use gdown if available
                    try:
                        import gdown
                        output = gdown.download(id=file_id, output=file_download_path, quiet=False)
                        if output:
                            success_count += 1
                    except:
                        # Fall back to direct download
                        url = f"https://drive.google.com/uc?id={file_id}&export=download"
                        response = session.get(url, stream=True)
                        
                        with open(file_download_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=4096):
                                if chunk:
                                    f.write(chunk)
                        success_count += 1
                except Exception as e:
                    print(f"Error downloading file: {e}")
            
            if success_count > 0:
                print(f"Successfully downloaded {success_count} out of {len(file_ids)} files.")
                return download_folder
            else:
                print("Failed to download any files from the folder.")
                return None
    except Exception as e:
        print(f"Error listing folder contents: {e}")
    
    # Method 3: Fall back to suggesting the user create a zip file
    print("\nCould not automatically download files from the Google Drive folder.")
    print("Please consider the following options:")
    print("1. Download the folder manually and provide a local folder path")
    print("2. Create a zip file of the folder in Google Drive, share it, and provide the direct link to the zip file")
    print("3. Install gdown (pip install gdown) and try again, as it has better folder download support")
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Modified parameters for local folder usage
    parser.add_argument('--real_folder', type=str, required=True, 
                        help='Local folder path or Google Drive link containing real images')
    parser.add_argument('--fake_folder', type=str, required=True, 
                        help='Local folder path or Google Drive link containing fake/generated images')
    parser.add_argument('--install_deps', action='store_true',
                        help='Automatically install dependencies like gdown if needed')
    parser.add_argument('--use_colab', action='store_true',
                        help='Use Google Colab authentication for Google Drive access')
    parser.add_argument('--max_sample', type=int, default=None, 
                        help='Limit samples for both fake/real (None=use all available images)')
    parser.add_argument('--arch', type=str, default='res50', 
                        help='Model architecture')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth', 
                        help='Path to model checkpoint')
    parser.add_argument('--result_folder', type=str, default='result', 
                        help='Folder to save results')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for evaluation')
    parser.add_argument('--jpeg_quality', type=int, default=None, 
                        help="100, 90, 80, ... 30. Used to test robustness. Not applied if None")
    parser.add_argument('--gaussian_sigma', type=float, default=None, 
                        help="0,1,2,3,4. Used to test robustness. Not applied if None")
    parser.add_argument('--temp_dir', type=str, default='./temp_downloads',
                        help='Directory to store downloaded files')

    opt = parser.parse_args()
    
    # Create result folder
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    # Load model
    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print("Model loaded..")
    model.eval()
    model.cuda()

    # Install dependencies if requested
    if opt.install_deps:
        try:
            print("Installing required dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "--upgrade"])
            print("Successfully installed dependencies")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
    
    # Google Colab authentication (if specified)
    if opt.use_colab:
        try:
            from google.colab import auth
            from google.colab import drive
            
            print("Authenticating with Google...")
            auth.authenticate_user()
            
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            
            # If the path is in Google Drive format but already mounted
            if opt.real_folder.startswith('drive/'):
                opt.real_folder = os.path.join('/content', opt.real_folder)
                print(f"Using mounted path for real images: {opt.real_folder}")
            if opt.fake_folder.startswith('drive/'):
                opt.fake_folder = os.path.join('/content', opt.fake_folder)
                print(f"Using mounted path for fake images: {opt.fake_folder}")
        except ImportError:
            print("Google Colab modules not found. Skipping authentication.")
        except Exception as e:
            print(f"Error authenticating with Google: {e}")
    
    # Process real folder (handle Google Drive links)
    local_real_folder = opt.real_folder
    if local_real_folder.startswith(('http://', 'https://')):
        print("Detected URL for real images. Attempting to download...")
        downloaded_folder = download_from_gdrive(local_real_folder, os.path.join(opt.temp_dir, "real"))
        if downloaded_folder is None:
            print("Error: Could not download or process the Google Drive URL for real images.")
            exit(1)
        local_real_folder = downloaded_folder
        print(f"Using downloaded files from: {local_real_folder}")
    else:
        # Check if it's a valid local path
        if not os.path.exists(local_real_folder):
            print(f"Error: The specified real folder path {local_real_folder} does not exist.")
            exit(1)
        print(f"Using local folder for real images: {local_real_folder}")
    
    # Process fake folder (handle Google Drive links)
    local_fake_folder = opt.fake_folder
    if local_fake_folder.startswith(('http://', 'https://')):
        print("Detected URL for fake images. Attempting to download...")
        downloaded_folder = download_from_gdrive(local_fake_folder, os.path.join(opt.temp_dir, "fake"))
        if downloaded_folder is None:
            print("Error: Could not download or process the Google Drive URL for fake images.")
            exit(1)
        local_fake_folder = downloaded_folder
        print(f"Using downloaded files from: {local_fake_folder}")
    else:
        # Check if it's a valid local path
        if not os.path.exists(local_fake_folder):
            print(f"Error: The specified fake folder path {local_fake_folder} does not exist.")
            exit(1)
        print(f"Using local folder for fake images: {local_fake_folder}")
        
    # Create the dataset with local folders
    set_seed()
    print("\nCreating dataset with local real and fake images...")
    
    dataset = LocalImageDataset(
        local_real_folder,
        local_fake_folder,
        opt.max_sample,
        opt.arch,
        jpeg_quality=opt.jpeg_quality,
        gaussian_sigma=opt.gaussian_sigma
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Get validation results
    print("\nEvaluating real vs fake images...")
    ap, r_acc0, f_acc0, acc0, precision0, recall0, f1_0, r_acc1, f_acc1, acc1, precision1, recall1, f1_1, best_thres = validate(model, loader, find_thres=True)

    # Create results dictionary
    results_dict = {
        'Model': 'local_evaluation',
        'AP': round(ap*100, 2),
        'Threshold_0.5': {
            'Real_Accuracy': round(r_acc0*100, 2),
            'Fake_Accuracy': round(f_acc0*100, 2),
            'Overall_Accuracy': round(acc0*100, 2),
            'Precision': round(precision0*100, 2),
            'Recall': round(recall0*100, 2),
            'F1_Score': round(f1_0*100, 2)
        },
        'Best_Threshold': {
            'Threshold': best_thres,
            'Real_Accuracy': round(r_acc1*100, 2),
            'Fake_Accuracy': round(f_acc1*100, 2),
            'Overall_Accuracy': round(acc1*100, 2),
            'Precision': round(precision1*100, 2),
            'Recall': round(recall1*100, 2),
            'F1_Score': round(f1_1*100, 2)
        }
    }

    # Write results
    with open(os.path.join(opt.result_folder, 'ap.txt'), 'a') as f:
        f.write(f"local_real vs local_fake: {round(ap*100, 2)}\n")

    with open(os.path.join(opt.result_folder, 'metrics_0.5.txt'), 'a') as f:
        f.write(f"local_real vs local_fake: Acc={round(acc0*100, 2)} Real={round(r_acc0*100, 2)} "
               f"Fake={round(f_acc0*100, 2)} Precision={round(precision0*100, 2)} "
               f"Recall={round(recall0*100, 2)} F1={round(f1_0*100, 2)}\n")
        
    with open(os.path.join(opt.result_folder, 'metrics_best.txt'), 'a') as f:
        f.write(f"local_real vs local_fake: Threshold={best_thres:.4f} Acc={round(acc1*100, 2)} "
               f"Real={round(r_acc1*100, 2)} Fake={round(f_acc1*100, 2)} "
               f"Precision={round(precision1*100, 2)} Recall={round(recall1*100, 2)} "
               f"F1={round(f1_1*100, 2)}\n")
    
    # Print detailed results
    print(f"\nResults for local_real vs local_fake:")
    print(f"AP: {round(ap*100, 2)}")
    print("\nWith threshold=0.5:")
    print(f"  Accuracy: {round(acc0*100, 2)}% (Real: {round(r_acc0*100, 2)}%, Fake: {round(f_acc0*100, 2)}%)")
    print(f"  Precision: {round(precision0*100, 2)}%")
    print(f"  Recall: {round(recall0*100, 2)}%")
    print(f"  F1 Score: {round(f1_0*100, 2)}%")
    
    print(f"\nWith best threshold={best_thres:.4f}:")
    print(f"  Accuracy: {round(acc1*100, 2)}% (Real: {round(r_acc1*100, 2)}%, Fake: {round(f_acc1*100, 2)}%)")
    print(f"  Precision: {round(precision1*100, 2)}%")
    print(f"  Recall: {round(recall1*100, 2)}%")
    print(f"  F1 Score: {round(f1_1*100, 2)}%")

    print("\nEvaluation completed!")