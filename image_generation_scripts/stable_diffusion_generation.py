import torch
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor 
import json
import requests 
import subprocess
import re
import argparse
import yaml
import os
import logging
import shutil
import numpy as np
from PIL import Image 
from tqdm import tqdm
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
from cleanfid import fid
import json
from torchvision.transforms.functional import pil_to_tensor
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image


model_id = 'stabilityai/stable-diffusion-xl-base-1.0'



# os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda')
print(device)


print("Setting up initial data...")

with open('captions_train2014_new_format.json','r') as file : 
    new = json.load(file)


with open('/raid/home/ashhar21137/watermarking/data_card_gen/id_fid_new.json', 'r') as prev_file:
    previous_data = json.load(prev_file)

# Initialize the processed_ids set with keys from the previous data
processed_ids = set(previous_data.keys())

print("1 : ",len(processed_ids))

# Load the additional data
with open('/raid/home/ashhar21137/watermarking/data_card_gen/id_fid.json', 'r') as additional_file:
    additional_data = json.load(additional_file)

# Update the processed_ids set with keys from the additional data
processed_ids.update(additional_data.keys())

print("2 : ",len(processed_ids))

print(f"processed_ids : {processed_ids}")


# processed_ids = list(processed_ids)


img_urls = dict()

def download_image(url, file_name, save_dir,img_urls):
    try:
        # Create save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Download the image
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the image to the specified directory
        with open(os.path.join(save_dir, file_name), 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
                
        return True
    except requests.RequestException as e:
        print(f"Failed to download from {url}: {e}")
        return False

save_dir = 'example_3_1k'

# Number of images to be tested
count = 2
for i in tqdm(range(len(new['images']))):
    id = str(new['images'][i]['id'])

    if id in processed_ids:
        print(f"Id : {id} will be skipped")
        continue

    file_name = f"Image_{new['images'][i]['id']}.jpg"
    img_urls[new['images'][i]['id']] = new['images'][i]['coco_url']
    
    if not download_image(new['images'][i]['coco_url'], file_name, save_dir, img_urls):
        if not download_image(new['images'][i]['flickr_url'], file_name, save_dir, img_urls):
            print("Failed to download the image from both URLs")
            continue      

    # print(new['annotations'][f'{id}'])

    count = count - 1 
    if(count == 0 ) : 
        break

print(f'Image urls : {img_urls}')


# Define the directory containing the images
directory = r'example_3_1k'

# Define the new size
new_size = (784, 784)

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add any other formats you need
        file_path = os.path.join(directory, filename)
        
        # Open an image file
        with Image.open(file_path) as img:
            # Resize image
            img_resized = img.resize(new_size, Image.LANCZOS)
            
            # Save it back to the same location
            img_resized.save(file_path)

print("All images have been resized successfully.")


def get_img_tensor(img_path, device):
    img_tensor = pil_to_tensor(Image.open(img_path).convert("RGB"))/255
    return img_tensor.unsqueeze(0).to(device)



img_dir = 'example_3_1k'
img_ids = []
img_pths = []
imgs = os.listdir(img_dir)
for i in imgs : 
    if '.jpg' in i :
        id = re.split('[_.]',i)[1]
        img_ids.append(id) 
        img_pths.append(os.path.join(img_dir,i))

print(f'Image ids : {img_ids}')

captions = []
for id in img_ids : 
    captions.append(new['annotations'][id])


print(captions)

print(f"ids : {img_ids}")
print(f"paths : {img_pths}")
print(f"captions : {captions}")
# print(f"prompt : {prompt}")


neg_prompt = 'deformity, bad anatomy, cloned face, amputee, people in background, asymmetric, disfigured, extra limbs, text, missing legs, missing arms, Out of frame, low quality, Poorly drawn feet'


input_dir = 'input_copies_3_1k'
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

print("Creating copies of input images")
for i in range(len(img_ids)):
    save_path = os.path.join(input_dir,f'{img_ids[i]}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for copy_count in range(5):
        filename = f'{copy_count}__image_{img_ids[i]}.png'
        img = Image.open(img_pths[i])
        img.save(os.path.join(save_path, filename))
        print(f"Image saved to {os.path.join(save_path, filename)}")

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False).to(device)
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)

id_fid = dict()

print(f'Starting visual paraphrasing')
for i in tqdm(range(len(img_ids))) : 
    ig = Image.open(img_pths[i])
    init_image = load_image(ig)
    # print(init_image)

    gen_image = pipeline(captions[i], image=init_image, strength=0.2, guidance_scale=7.5).images

    directory =  f'generated_images_3_1k/{img_ids[i]}'
    print(f"Saving generated images at {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    for k in range(len(gen_image)) :
        gen_save_dir = os.path.join(directory,f'{img_ids[i]}_gen_{k}.png')
        gen_image[k].save(gen_save_dir)
        print(f"Generated Image saved to {gen_save_dir}")


    print(f'FID scoring between images saved at input_copies_3_1k/{img_ids[i]} and generated_images_3_1k/{img_ids[i]}')
    score = fid.compute_fid(f'input_copies_3_1k/{img_ids[i]}', f'generated_images_3_1k/{img_ids[i]}')
    id_fid[img_ids[i]] = dict()
    id_fid[img_ids[i]]['FID_score'] = score
    id_fid[img_ids[i]]['captions'] = captions[i]


    print(img_ids[i])
    print(img_urls[int(img_ids[i])])
    id_fid[img_ids[i]]['url'] = img_urls[int(img_ids[i])]


    print(f'FID score for image id {id} : {score}') 
    print("---------------------------------------------")
    
    input_copies_2_dir = f'input_copies_3_1k/{img_ids[i]}'
    if os.path.exists(input_copies_2_dir):
        shutil.rmtree(input_copies_2_dir)
        print(f"Deleted folder {input_copies_2_dir}")


print(f'Saving json file')
with open("id_fid_new_1k.json", "w") as json_file:
    json.dump(id_fid, json_file, indent=4)