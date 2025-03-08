#!/bin/bash
#sbatch --get-user-env=L                #replicate login env

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=de-fake_infer_coco_sd35      #Set the job name to "JobExample4"
#SBATCH --time=1-00:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                #Request 1 node
#SBATCH --ntasks-per-node=1        #Request 8 tasks/cores per node
#SBATCH --mem=32G                     #Request 16GB per node
#SBATCH --output=de-fake_infer_coco_sd35.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:rtx:1          #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132705883597           #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=hasnat.md.abdullah@tamu.edu    #Send all emails to email_address



# run aide inference on coco images
# eval data path: the dir should have two subfolders: 0_real and 1_fake
# each folder should have images: image_0.jpg (example)
# checkpoint="/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/aide/checkpoint/sd14_train.pth"

# coco
# eval_data_path="/scratch/user/hasnat.md.abdullah/VCT_2/data/aide_coco/"
# twitter
# eval_data_path="/scratch/user/hasnat.md.abdullah/VCT_2/data/aide_twitter/"
# edit and add the data img folder name (i.e sd35_coco_image) in line: 328 of inference file
# python aide_inference.py --eval_data_path $eval_data_path --resume $checkpoint

#--------------------------------------------
# run deepfake inference on twitter sd3.5 images
## 0.1 Feature Extraction
# cd /scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/deepfakedetection
# data_dir="/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_twitter/twitter_sd35/"
# python Feature_extraction_pretrained.py --data_dir $data_dir
## 0.2 inference
# cd /scratch/user/hasnat.md.abdullah/VCT_2/detection_scripts
# make sure that the last folder name of data dir (i.e twitter_sd35) is appended after pretrained_{folder_name}.csv
# x_test='/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/deepfakedetection/X_test_pretrained_twitter_sd35.csv'
# y_test='/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/deepfakedetection/y_test_pretrained_twitter_sd35.csv'
# python deepfake_detection_inference.py --x_test $x_test --y_test $y_test


# run deepfake inference on coco sd3.5 images
## 0.1 Feature Extraction
# cd /scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/deepfakedetection
# data_dir="/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_coco/sd35_coco_image/"
# python Feature_extraction_pretrained.py --data_dir $data_dir
# ## 0.2 inference
# cd /scratch/user/hasnat.md.abdullah/VCT_2/detection_scripts
# # make sure that the last folder name of data dir (i.e sd35_coco_image) is appended after pretrained_{folder_name}.csv
# x_test='/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/deepfakedetection/X_test_pretrained_sd35_coco_image.csv'
# y_test='/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/deepfakedetection/y_test_pretrained_sd35_coco_image.csv'
# python deepfake_detection_inference.py --x_test $x_test --y_test $y_test

# run universal fake detect inference on coco-twitter sd3.5 images
##coco
# python universal_fake_detect_validate_sd35_new.py\
#  --real_folder '/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_coco/sd35_coco_image/original'\
#  --fake_folder "/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_coco/sd35_coco_image/fake"\
#  --arch CLIP:ViT-L/14 \
#  --ckpt '/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/UniversalFakeDetect/pretrained_weights/fc_weights.pth'\
#  --result_folder clip_vitl14_results 

## twitter
# python universal_fake_detect_validate_sd35_new.py\
#  --real_folder '/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_twitter/twitter_sd35/original'\
#  --fake_folder "/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_twitter/twitter_sd35/fake"\
#  --arch CLIP:ViT-L/14 \
#  --ckpt '/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/UniversalFakeDetect/pretrained_weights/fc_weights.pth'\
#  --result_folder clip_vitl14_results 

# run C2P-CLIP-DeepfakeDetection inference on coco-twitter sd3.5 images
##coco
# python c2p_clip_inference.py \
#   --real_folder '/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_coco/sd35_coco_image/original' \
#   --sd35_folder "/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_coco/sd35_coco_image/fake"
## twitter
# python c2p_clip_inference.py \
#   --real_folder '/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_twitter/twitter_sd35/original' \
#   --sd35_folder "/scratch/user/hasnat.md.abdullah/VCT_2/data/deepfake_twitter/twitter_sd35/fake"

#twitter
python de-fake_inference.py