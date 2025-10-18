
: '
This is the demo for shanghai dataset.
To use on a new dataset, you need to edit lines position of 'edit here' RainDiff/datasets/get_datasets.py for your new dataset and define its class in RainDiff/datasets/dataset_your_dataset_name.py
'

cd 'your_dir'/RainDiff/resources/data
wget https://huggingface.co/datasets/Thao123/weather/resolve/main/shanghai.h5

cd 'your_dir'/RainDiff
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


: '
where:
    backbone: deterministic backbone, default: simvp, dont change,
    dataset: your_dataset_name,
    use_diff: use diffusion or not, dont change,
    batch_size: default: 20,
    frames_in: Number of input frames, default: 5,
    frames_out: Number of output frames, default: 20,
    training_steps: Number of training step, default: 200000,

In case of evaluation, add:
    --eval: evaluation mode
    --ckpt_milestone: your pretrained weight

'

