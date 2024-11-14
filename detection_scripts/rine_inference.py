from src.utils import get_transforms, get_our_trained_model
from PIL import Image
import torch
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

# Set device
device = "cuda"
print("Device set to:", device)

# Load model and transformations
print("Loading model and transformations...")
_, transforms, _ = get_transforms()
model = get_our_trained_model(ncls="ldm", device=device)
model.to(device)
model.eval()
print("Model loaded and set to evaluation mode.")

# Define image paths
real_images_path = "D:\\path_to_dataset\\saved_images_twitter\\twitter_image"
generated_images_path = "D:\\path_to_dataset\\saved_images_twitter\\dalle_image"
print(f"Real images path: {real_images_path}")
print(f"Generated images path: {generated_images_path}")

# Output CSV file path
output_csv_path = "t_rine_detection_results_dalle.csv"
print(f"Output CSV path: {output_csv_path}")

# Function to get the probability of being "fake"
def get_fake_probability(image_path):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms(image).unsqueeze(0).to(device)
    logit = model(image_tensor)[0]
    probability = torch.sigmoid(logit).detach().cpu().numpy()[0][0]
    print(f"Probability of being fake: {probability:.4f}")
    return probability

# Write predictions to CSV
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "true_label", "predicted_label", "probability"])
    print("CSV file created, writing headers...")

    # Evaluate real images (label 0)
    print("Evaluating real images...")
    for filename in os.listdir(real_images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(real_images_path, filename)
            prob_fake = get_fake_probability(image_path)
            predicted_label = 1 if prob_fake > 0.5 else 0
            writer.writerow([filename, 0, predicted_label, prob_fake])
            print(f"Saved result for real image: {filename}")

    # Evaluate generated images (label 1)
    print("Evaluating generated images...")
    for filename in os.listdir(generated_images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(generated_images_path, filename)
            prob_fake = get_fake_probability(image_path)
            predicted_label = 1 if prob_fake > 0.5 else 0
            writer.writerow([filename, 1, predicted_label, prob_fake])
            print(f"Saved result for generated image: {filename}")

print("All images processed and results saved to CSV.")

# Load CSV and calculate metrics
print("Calculating metrics from CSV...")
true_labels = []
predicted_labels = []

with open(output_csv_path, mode="r") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        true_labels.append(int(row["true_label"]))
        predicted_labels.append(int(row["predicted_label"]))

# Calculate Precision, Recall, and F1-score
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("Evaluation metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

