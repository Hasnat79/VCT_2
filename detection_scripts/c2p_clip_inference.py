import sys
sys.path.append('/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/C2P-CLIP-DeepfakeDetection')
import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
import zipfile
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import warnings
import random
from tqdm import tqdm
import json
import glob

warnings.filterwarnings('ignore')

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
seed_torch(123)

class C2P_CLIP(nn.Module):
    def __init__(self, name='openai/clip-vit-large-patch14', num_classes=1):
        super(C2P_CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained(name)
        del self.model.text_model
        del self.model.text_projection
        del self.model.logit_scale
        
        self.model.vision_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.fc = nn.Linear(768, num_classes)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

    def encode_image(self, img):
        vision_outputs = self.model.vision_model(
            pixel_values=img,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.use_return_dict,      
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features    

    def forward(self, img):
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return self.model.fc(image_embeds)
    
    def get_features(self, img):
        """Get image features for decoding"""
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds

# Feature decoding functions
def get_clip_model(clip_name, device):
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name)
    clip_model = clip_model.to(device)
    clip_model.eval()
    return clip_model, processor

def get_clipcap_model(model_path, device):
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Load the model weights
        if model_path.startswith('http'):
            state_dict = torch.hub.load_state_dict_from_url(model_path, map_location=device)
        else:
            state_dict = torch.load(model_path, map_location=device)
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading CLIPCap model: {e}")
        return None, None

def get_text(image_features, tokenizer, model, fc_path, cal_detection_feat=False, device="cuda:0"):
    try:
        # Load FC parameters
        if fc_path.startswith('http'):
            fc_params = torch.hub.load_state_dict_from_url(fc_path, map_location=device)
        else:
            fc_params = torch.load(fc_path, map_location=device)
        
        fc = torch.nn.Linear(image_features.shape[-1], model.transformer.wte.weight.shape[1]).to(device)
        fc.load_state_dict(fc_params)
        
        # Generate text with features
        image_features = image_features.to(device)
            
        prefix = fc(image_features).view(1, 10, -1)
        
        generated_text_prefix = model.generate(
            input_ids=None,
            max_length=40,
            min_length=5,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            prefix_embeds=prefix
        )
        
        generated_text = tokenizer.decode(generated_text_prefix[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error generating text: {e}")
        return "Error in text generation"

class LocalImageDataset(Dataset):
    def __init__(self, real_folder, fake_folder, transform=None):
        """
        Dataset for comparing real images with AI generated images from local folders.
        
        Args:
            real_folder: Path to folder containing real images
            fake_folder: Path to folder containing fake/generated images
            transform: Image transformations to apply
        """
        self.transform = transform
        
        # Get all image files from both folders
        self.real_image_paths = self._get_image_files(real_folder)
        self.fake_image_paths = self._get_image_files(fake_folder)
        
        # Use the minimum number of images from both folders to ensure balanced dataset
        self.num_images = min(len(self.real_image_paths), len(self.fake_image_paths))
        
        # Trim lists to the same length
        self.real_image_paths = self.real_image_paths[:self.num_images]
        self.fake_image_paths = self.fake_image_paths[:self.num_images]
        
        print(f"Found {self.num_images} images in each folder")
        
    def _get_image_files(self, folder):
        # Get all image files in the folder
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder, ext)))
            image_paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
            
        return sorted(image_paths)
        
    def __len__(self):
        return self.num_images * 2  # Each pair gives one real and one fake sample
    
    def __getitem__(self, idx):
        is_fake = idx % 2  # 0 for real, 1 for fake
        pair_idx = idx // 2
        
        # Generate a dummy caption based on the filename
        if is_fake:
            image_path = self.fake_image_paths[pair_idx]
            label = 1  # fake/generated
            image_id = f"{pair_idx}_fake"
            caption = f"Generated image: {os.path.basename(image_path)}"
        else:
            image_path = self.real_image_paths[pair_idx]
            label = 0  # real
            image_id = f"{pair_idx}_real"
            caption = f"Real image: {os.path.basename(image_path)}"
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        # Apply transformations if available
        if self.transform:
            image = self.transform(image)
            
        return image, label, image_id, caption
    
    def get_sample_pair(self, idx):
        """Get a pair of real and fake images for feature analysis"""
        if idx >= self.num_images:
            idx = self.num_images - 1
            
        real_path = self.real_image_paths[idx]
        fake_path = self.fake_image_paths[idx]
        
        return {
            'real_image_path': real_path,
            'fake_image_path': fake_path,
            'caption': f"Image pair {idx}"
        }

def create_dataloader(real_folder, fake_folder, batch_size=32, loadSize=224, cropSize=224, no_resize=False, no_crop=False):
    # Define image transformations
    transform_list = []
    
    if not no_resize:
        transform_list.append(transforms.Resize(loadSize))
    if not no_crop:
        transform_list.append(transforms.CenterCrop(cropSize))
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    
    transform = transforms.Compose(transform_list)
    
    # Create dataset
    local_dataset = LocalImageDataset(
        real_folder=real_folder,
        fake_folder=fake_folder,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        local_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return dataloader, local_dataset

def print_section(section_name):
    print("=" * 60)
    print(f"{section_name.center(60)}")
    print("=" * 60)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate C2P-CLIP on local image folders')
    
    # Input options
    parser.add_argument('--real_folder', type=str, required=True, help='Path to real images folder')
    parser.add_argument('--sd35_folder', type=str, required=True, help='Path to SD3.5 generated images folder')
    
    # Model parameters
    parser.add_argument('--loadSize', type=int, default=224)
    parser.add_argument('--cropSize', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip')
    # parser.add_argument('--model_path', type=str, default='/scratch/user/hasnat.md.abdullah/VCT_2/detection_scripts/cache/C2P_CLIP_release_20240901.zip')
    parser.add_argument('--clipcap_model_path', type=str, default='https://www.now61.com/f/Xljmi0/coco_prefix_latest.pt')
    # parser.add_argument('--clipcap_model_path', type=str, default='/scratch/user/hasnat.md.abdullah/VCT_2/detection_scripts/cache/coco_prefix_latest.pt')
    parser.add_argument('--fc_path', type=str, default='https://www.now61.com/f/qwvoH5/fc_parameters.pth')
    # parser.add_argument('--fc_path', type=str, default='/scratch/user/hasnat.md.abdullah/VCT_2/detection_scripts/cache/fc_parameters.pth')
    parser.add_argument('--save_path', type=str, default='local_results')
    parser.add_argument('--no_resize', action='store_true')
    parser.add_argument('--no_crop', action='store_true')
    parser.add_argument('--decode_features', action='store_true')
    parser.add_argument('--sample_count', type=int, default=50)
    
    args = parser.parse_args()
    
    # Print arguments
    print('----------------- Options ---------------')
    for k, v in sorted(vars(args).items()):
        print(f'{k:>25}: {v:<30}')
    print('----------------- End -------------------')
    
    return args

def analyze_features(dataset, opt, model, clipcap_model=None, clipcap_tokenizer=None, 
                    generator_name='generated', device='cuda'):
    """Analyze image features for a sample of images and decode to text if requested"""
    
    # Create output directory
    analysis_dir = os.path.join(opt.save_path, f"real_vs_{generator_name}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Sample a subset of examples
    sample_size = min(opt.sample_count, dataset.num_images)
    sample_indices = random.sample(range(dataset.num_images), sample_size)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(opt.loadSize) if not opt.no_resize else transforms.Lambda(lambda x: x),
        transforms.CenterCrop(opt.cropSize) if not opt.no_crop else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    results = []
    
    for idx in tqdm(sample_indices, desc=f"Analyzing {generator_name}"):
        example = dataset.get_sample_pair(idx)
        caption = example['caption']
        
        # Load images
        try:
            real_img = Image.open(example['real_image_path']).convert('RGB')
            fake_img = Image.open(example['fake_image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            continue
        
        # Apply transformations
        real_tensor = transform(real_img).unsqueeze(0).to(device)
        fake_tensor = transform(fake_img).unsqueeze(0).to(device)
        
        # Get predictions and features
        with torch.no_grad():
            real_score = model(real_tensor).sigmoid().item()
            fake_score = model(fake_tensor).sigmoid().item()
            
            real_features = model.get_features(real_tensor)
            fake_features = model.get_features(fake_tensor)
            
            # Compute feature similarity
            similarity = torch.nn.functional.cosine_similarity(real_features, fake_features).item()
            
            # Get feature descriptions if requested
            real_description = None
            fake_description = None
            
            if opt.decode_features and clipcap_model is not None and clipcap_tokenizer is not None:
                real_description = get_text(
                    real_features,
                    clipcap_tokenizer,
                    clipcap_model,
                    opt.fc_path,
                    cal_detection_feat=False,
                    device=device
                )
                
                fake_description = get_text(
                    fake_features,
                    clipcap_tokenizer,
                    clipcap_model,
                    opt.fc_path,
                    cal_detection_feat=False,
                    device=device
                )
        
        # Store results
        result = {
            'example_id': idx,
            'real_image_path': example['real_image_path'],
            'fake_image_path': example['fake_image_path'],
            'caption': caption,
            'real_score': real_score,
            'fake_score': fake_score,
            'similarity': similarity
        }
        
        if real_description:
            result['real_description'] = real_description
        if fake_description:
            result['fake_description'] = fake_description
            
        results.append(result)
    
    # Save results
    with open(os.path.join(analysis_dir, 'feature_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate aggregate statistics
    avg_similarity = np.mean([r['similarity'] for r in results])
    avg_real_score = np.mean([r['real_score'] for r in results])
    avg_fake_score = np.mean([r['fake_score'] for r in results])
    
    print(f"\nFeature Analysis Summary for real vs {generator_name}:")
    print(f"Average Feature Similarity: {avg_similarity:.4f}")
    print(f"Average Real Score: {avg_real_score:.4f}")
    print(f"Average Fake Score: {avg_fake_score:.4f}")
    
    return results

if __name__ == '__main__':
    opt = parse_args()
    
    # Create output directory
    os.makedirs(opt.save_path, exist_ok=True)
    
    # Load C2P-CLIP model
    print("Loading C2P-CLIP model...")
    state_dict = torch.hub.load_state_dict_from_url(opt.model_path, map_location="cpu", progress=True)
    model = C2P_CLIP(name='openai/clip-vit-large-patch14', num_classes=1)
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Load CLIPCap model for feature decoding if requested
    clipcap_model = None
    clipcap_tokenizer = None
    
    if opt.decode_features:
        print("Loading CLIPCap model for feature decoding...")
        clipcap_model, clipcap_tokenizer = get_clipcap_model(opt.clipcap_model_path, device)
        if clipcap_model is None:
            print("Warning: CLIPCap model could not be loaded. Feature decoding will be disabled.")
            opt.decode_features = False
    
    # Store results for all generators
    all_results = {}
    feature_analysis_results = {}
    
    # Evaluate SD3.5 images against real images
    print_section(f"Evaluating real vs SD3.5")
    
    # Check if folders exist
    if not os.path.exists(opt.real_folder) or not os.path.isdir(opt.real_folder):
        print(f"Error: Real folder '{opt.real_folder}' does not exist or is not a directory")
        exit(1)
        
    if not os.path.exists(opt.sd35_folder) or not os.path.isdir(opt.sd35_folder):
        print(f"Error: SD3.5 folder '{opt.sd35_folder}' does not exist or is not a directory")
        exit(1)
    
    # Create dataloader
    data_loader, dataset = create_dataloader(
        real_folder=opt.real_folder, 
        fake_folder=opt.sd35_folder,
        batch_size=opt.batch_size,
        loadSize=opt.loadSize,
        cropSize=opt.cropSize,
        no_resize=opt.no_resize,
        no_crop=opt.no_crop
    )
        
    # Run inference
    with torch.no_grad():
        y_true, y_pred, image_ids, captions = [], [], [], []
        
        for batch_images, batch_labels, batch_ids, batch_captions in tqdm(data_loader, desc="Evaluating SD3.5"):
            batch_preds = model(batch_images.to(device)).sigmoid().flatten().cpu().numpy()
            
            y_pred.extend(batch_preds.tolist())
            y_true.extend(batch_labels.numpy().tolist())
            image_ids.extend(batch_ids)
            captions.extend(batch_captions)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    ap = average_precision_score(y_true, y_pred)
    
    # Calculate metrics for real and fake images separately
    real_indices = (y_true == 0)
    if np.any(real_indices):
        real_acc = accuracy_score(y_true[real_indices], y_pred_binary[real_indices])
        print(f"Real images accuracy: {real_acc:.4f}")
        
    fake_indices = (y_true == 1)
    if np.any(fake_indices):
        fake_acc = accuracy_score(y_true[fake_indices], y_pred_binary[fake_indices])
        print(f"Generated images accuracy: {fake_acc:.4f}")
    
    # Store detailed results
    detailed_results = []
    for i in range(len(y_true)):
        result = {
            'image_id': image_ids[i],
            'caption': captions[i],
            'true_label': int(y_true[i]),  # Convert to Python int
            'predicted_score': float(y_pred[i]),  # Convert to Python float
            'predicted_label': int(y_pred_binary[i]),  # Convert to Python int
            'correct': bool(y_true[i] == y_pred_binary[i])  # Convert to Python bool
        }
        detailed_results.append(result)
        
    # Save detailed results
    results_dir = os.path.join(opt.save_path, "real_vs_sd35")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Store summary metrics - convert all values to Python native types
    results_summary = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'ap': float(ap),
        'real_accuracy': float(real_acc) if np.any(real_indices) else None,
        'fake_accuracy': float(fake_acc) if np.any(fake_indices) else None
    }
    
    # Print results
    print(f"Results for real vs SD3.5:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Analyze features for a subset of examples
    if opt.sample_count > 0:
        print(f"\nAnalyzing features for SD3.5...")
        feature_analysis = analyze_features(
            dataset, 
            opt, 
            model, 
            clipcap_model=clipcap_model, 
            clipcap_tokenizer=clipcap_tokenizer,
            generator_name="sd35", 
            device=device
        )
    
    # Save summary metrics to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AP', 'Real Accuracy', 'Fake Accuracy'],
        'Value': [
            results_summary['accuracy'],
            results_summary['precision'],
            results_summary['recall'],
            results_summary['f1_score'],
            results_summary['ap'],
            results_summary['real_accuracy'] if results_summary['real_accuracy'] is not None else None,
            results_summary['fake_accuracy'] if results_summary['fake_accuracy'] is not None else None
        ]
    })
    
    results_df.to_csv(os.path.join(opt.save_path, 'detection_metrics.csv'), index=False)