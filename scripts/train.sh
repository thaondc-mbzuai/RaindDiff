cd ./resources/data
wget https://huggingface.co/datasets/Thao123/RainDiff/resolve/main/shanghai.h5

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
  --training_steps 300000 \
  --exp_note raindiff
