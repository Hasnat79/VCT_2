import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as tt

class task1Dataset(Dataset): 
    def __init__(self, data_paths, labels, transform):
        print("Initializing dataset with", len(data_paths), "images")
        self.data = [Image.open(img).convert("RGB") for img in tqdm(data_paths)]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.data)

def load_data_paths_and_labels(real_image_path, generated_image_path):
    print("Loading data paths and labels...")
    real_image_paths = [os.path.join(real_image_path, img) for img in os.listdir(real_image_path) if img.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    generated_image_paths = [os.path.join(generated_image_path, img) for img in os.listdir(generated_image_path) if img.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    data_paths = real_image_paths + generated_image_paths
    labels = [0] * len(real_image_paths) + [1] * len(generated_image_paths)
    print("Data paths and labels loaded.")
    return data_paths, labels

def evaluate_model_on_dataset(model, criterion, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    return auc, accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on concatenated dataset")
    parser.add_argument("--real_image_path", type=str, required=True, help="Path to the directory of real images")
    parser.add_argument("--generated_image_path", type=str, required=True, help="Path to the directory of generated images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_path", type=str, default="results.txt", help="Path to save the results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path, map_location=device)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define transformations
    transform = tt.Compose([tt.Resize((256, 256)), tt.ToTensor(), tt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Load data paths and labels
    data_paths, labels = load_data_paths_and_labels(args.real_image_path, args.generated_image_path)

    # Create dataset and DataLoader
    dataset = task1Dataset(data_paths, labels, transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate model
    auc, accuracy = evaluate_model_on_dataset(model, criterion, data_loader, device)

    # Save results
    with open(args.output_path, 'a') as f:
        f.write(f"Test AUC: {auc:.4f}, Accuracy: {accuracy:.4f}\n")
    print("Results saved to", args.output_path)

if __name__ == "__main__":
    main()
