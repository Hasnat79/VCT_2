import os
from collections import OrderedDict
import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import albumentations as A
import albumentations.pytorch as Ap
from utils import architectures
from utils.python_patch_extractor.PatchExtractor import PatchExtractor
from PIL import Image
import argparse

class Detector:
    def __init__(self):
        self.weights_path_list = [os.path.join('weights', f'method_{x}.pth') for x in 'ABCDE']

        # GPU configuration if available
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []
        for i, l in enumerate('ABCDE'):
            network_class = getattr(architectures, 'EfficientNetB4')
            net = network_class(n_classes=2, pretrained=False).eval().to(self.device)
            print(f'Loading model {l}...')
            state_tmp = torch.load(self.weights_path_list[i], map_location='cpu')

            if 'net' not in state_tmp.keys():
                state = OrderedDict({'net': OrderedDict()})
                [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
            else:
                state = state_tmp
            incomp_keys = net.load_state_dict(state['net'], strict=True)
            print(incomp_keys)
            print('Model loaded!\n')

            self.nets += [net]

        net_normalizer = net.get_normalizer()  # pick normalizer from last network
        transform = [
            A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),
            Ap.transforms.ToTensorV2()
        ]
        self.trans = A.Compose(transform)
        self.cropper = A.RandomCrop(width=128, height=128, always_apply=True, p=1.)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def synth_real_detector(self, img_path: str, n_patch: int = 200):
        # Load image:
        img = np.asarray(Image.open(img_path))

        # Opt-out if image is non conforming
        if img.shape == ():
            print('{} None dimension'.format(img_path))
            return None
        if img.shape[0] < 128 or img.shape[1] < 128:
            print('Too small image')
            return None
        if img.ndim != 3:
            print('RGB images only')
            return None
        if img.shape[2] > 3:
            print('Omitting alpha channel')
            img = img[:, :, :3]

        print(f'Computing scores for {img_path}...')
        img_net_predictions = []
        for net_idx, net in enumerate(self.nets):

            if net_idx == 0:
                patch_list = [self.cropper(image=img)['image'] for _ in range(n_patch)]
            else:
                stride_0 = ((((img.shape[0] - 128) // 20) + 7) // 8) * 8
                stride_1 = (((img.shape[1] - 128) // 10 + 7) // 8) * 8
                pe = PatchExtractor(dim=(128, 128, 3), stride=(stride_0, stride_1, 3))
                patches = pe.extract(img)
                patch_list = list(patches.reshape((patches.shape[0] * patches.shape[1], 128, 128, 3)))

            # Normalization
            transf_patch_list = [self.trans(image=patch)['image'] for patch in patch_list]

            # Compute scores
            transf_patch_tensor = torch.stack(transf_patch_list, dim=0).to(self.device)
            with torch.no_grad():
                patch_scores = net(transf_patch_tensor).cpu().numpy()

                # Use argmax to determine the predicted class for each patch (0 for real, 1 for fake)
                patch_predictions = np.argmax(patch_scores, axis=1)
                
                # Majority voting: if any patch is classified as fake, classify the image as fake
                img_net_predictions.append(np.any(patch_predictions).astype(int))

        # Majority voting across the 5 networks
        final_prediction = np.round(np.mean(img_net_predictions)).astype(int)

        return final_prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='/path/to/your/dataset/Twitter_test/DALLE3/test', required=True)
    args = parser.parse_args()

    img_dir = args.img_dir

    detector = Detector()
    true_labels = []
    pred_labels = []

    # Traverse subdirectories like 'Original' and 'Fake'
    for root, dirs, files in os.walk(img_dir):
        for img_file in files:
            img_path = os.path.join(root, img_file)
            
            if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Define the truth label based on the folder name
                if 'Original' in root:
                    true_label = 0  # Real image
                elif 'Fake' in root:
                    true_label = 1  # Fake image
                else:
                    continue  # Skip files not in 'Original' or 'Fake' directories
                
                try:
                    # Get the final prediction from the detector
                    pred_label = detector.synth_real_detector(img_path)
                    
                    # Append the true and predicted labels for metric calculations
                    pred_labels.append(pred_label)
                    true_labels.append(true_label)

                except ValueError as e:
                    print(f"ValueError for file {img_path}: {e}")
                except Exception as e:
                    print(f"An error occurred with file {img_path}: {e}")

    valid_indices = [i for i, pred in enumerate(pred_labels) if pred is not None]
    true_labels_filtered = [true_labels[i] for i in valid_indices]
    pred_labels_filtered = [pred_labels[i] for i in valid_indices]

    if true_labels_filtered and pred_labels_filtered:
        accuracy = accuracy_score(true_labels_filtered, pred_labels_filtered)
        precision = precision_score(true_labels_filtered, pred_labels_filtered, pos_label=1)
        recall = recall_score(true_labels_filtered, pred_labels_filtered, pos_label=1)
        f1 = f1_score(true_labels_filtered, pred_labels_filtered, pos_label=1)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
    else:
        print('No valid predictions found for metric calculation.')
    
    return 0

if __name__ == '__main__':
    main()
