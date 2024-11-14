from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import itertools
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split
from torch import nn
from torchvision import transforms
import sys
import argparse
import time
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve
from blipmodels import blip_decoder
import glob

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

def preprocess_image(img_path, image_size=224):
    img = Image.open(img_path)
    img = img.resize((image_size, image_size))
    return preprocess(img)

parser = argparse.ArgumentParser(description='Finetune the classifier to wash the backdoor')
parser.add_argument('--real_dir', default='real_images', type=str, help='Directory containing real images')
parser.add_argument('--fake_dir', default='fake_images', type=str, help='Directory containing fake images')
parser.add_argument('--output_path', default='evaluation_results.txt', type=str, help='Path to save evaluation results')
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-B/32")

image_size = 224

blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

blip = blip_decoder(pretrained=blip_url, image_size=image_size, vit='base')
blip.eval()
blip = blip.to(device)

# Process all images from both directories
real_images = glob.glob(os.path.join(args.real_dir, '*'))
fake_images = glob.glob(os.path.join(args.fake_dir, '*'))
all_images = real_images + fake_images
true_labels = [0] * len(real_images) + [1] * len(fake_images)  # 0 for real, 1 for fake
predictions = []

# Load the models
model = torch.load("finetune_clip.pt").to(device)
linear = NeuralNet(1024,[512,256],2).to(device)
linear = torch.load('clip_linear.pt')

print(f"Processing {len(all_images)} images...")

for img_path in tqdm(all_images):
    try:
        img = Image.open(img_path).convert('RGB')
        tform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        img = tform(img)
        img = img.unsqueeze(0).to(device)

        # Generate caption using BLIP
        caption = blip.generate(img, sample=False, num_beams=3, max_length=60, min_length=5)
        text = clip.tokenize(list(caption)).to(device)

        # Process image through CLIP
        image = preprocess_image(img_path, image_size).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            emb = torch.cat((image_features, text_features), 1)
            output = linear(emb.float())
            predict = output.argmax(1)
            predictions.append(predict.cpu().numpy()[0])
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        continue

# Calculate metrics
predictions = np.array(predictions)
true_labels = np.array(true_labels[:len(predictions)])  # Adjust true_labels in case some images failed

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Print results to console
print(f"\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save results to text file
with open(args.output_path, 'w') as f:
    f.write("Evaluation Results\n")
    f.write("=================\n\n")
    f.write(f"Number of Real Images: {len(real_images)}\n")
    f.write(f"Number of Fake Images: {len(fake_images)}\n")
    f.write(f"Total Images Processed: {len(predictions)}\n\n")
    f.write("Metrics:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    
    # Add confusion matrix to text file
    f.write("Confusion Matrix:\n")
    cm = confusion_matrix(true_labels, predictions)
    f.write("            Predicted Real  Predicted Fake\n")
    f.write(f"Actual Real     {cm[0][0]:<14d} {cm[0][1]}\n")
    f.write(f"Actual Fake     {cm[1][0]:<14d} {cm[1][1]}\n")

print(f"\nEvaluation results saved to: {args.output_path}")

# Create and save confusion matrix visualization
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Real', 'Fake']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add text annotations to confusion matrix
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()