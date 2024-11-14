MODEL_NAME=convnext_base_in22k # Use clip-ViT-L-14 for UnivFD
MODEL_PATH=/home/muj/Documents/abhilekh/DRCT/pretrained/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth # Use ../output/pretrained/DRCT-2M/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth for UnivB
DEVICE_ID=0
EMBEDDING_SIZE=1024
MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${2:-$MODEL_PATH}
DEVICE_ID=${3:-$DEVICE_ID}
EMBEDDING_SIZE=${4:-$EMBEDDING_SIZE}
ROOT_PATH=/home/muj/Documents/abhilekh/saved_images/coco_image
FAKE_ROOT_PATH=/home/muj/Documents/abhilekh/saved_images/midjourney_image
DATASET_NAME=DRCT-2M
SAVE_TXT=/home/muj/Documents/abhilekh/DRCT/outputs/UnivB-midjourney-COCO.txt
INPUT_SIZE=224
BATCH_SIZE=24
FAKE_INDEXES=(1)
for FAKE_INDEX in ${FAKE_INDEXES[@]}
do
  echo FAKE_INDEX:${FAKE_INDEX}
  python train.py --root_path ${ROOT_PATH} --fake_root_path ${FAKE_ROOT_PATH} --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
done