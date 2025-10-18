cd ./resources/data
wget https://huggingface.co/datasets/Thao123/RainDiff/resolve/main/shanghai.h5

cd ../weights
wget https://huggingface.co/datasets/Thao123/RainDiff/resolve/main/model_shanghai.pt

cd ../..

conda env create -f env.ymal
conda activate raindiff

CUDA_VISIBLE_DEVICES=0 python run.py \
  --backbone simvp \
  --dataset shanghai \
  --use_diff \
  --batch_size 4 \
  --frames_in 5 \
  --frames_out 20 \
  --eval \
  --ckpt_milestone ./resources/weights/model_shanghai.pt \
  --exp_note raindiff
