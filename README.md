# DiffCast-CVPR2024
Official implementation of "[RainDiff: End to End Precipitation Nowcasting Via Token-wise Attention Diffusion](https://arxiv.org/pdf/2510.14962)"

![](resources/architecture.png)

## Abstract

Precipitation nowcasting, predicting future radar echo sequences from current observations, is a critical yet challenging task due to the inherently chaotic and tightly coupled spatio-temporal dynamics of the atmosphere. While recent advances in diffusion-based models attempt to capture both large-scale motion and fine-grained stochastic variability, they often suffer from scalability issues: latent-space approaches require a separately trained autoencoder, adding complexity and limiting generalization, while pixel-space approaches are computationally intensive and often omit attention mechanisms, reducing their ability to model long-range spatio-temporal dependencies. To address these limitations, we propose a Token-wise Attention integrated into not only the U-Net diffusion model but also the spatio-temporal encoder that dynamically captures multi-scale spatial interactions and temporal evolution. Unlike prior approaches, our method natively integrates attention into the architecture without incurring the high resource cost typical of pixel-space diffusion, thereby eliminating the need for separate latent modules. Our extensive experiments and visual evaluations across diverse datasets demonstrate that the proposed method significantly outperforms state-of-the-art approaches, yielding superior local fidelity, generalization, and robustness in complex precipitation forecasting scenarios.
## Code

### Environment

```shell
conda env create -f env.ymal
conda activate raindiff
```
### Resource
Dataset [Shanghai_Radar](https://drive.google.com/file/d/14JB4ElkZKHzqxIGKMFrbnY2P4zcae8RA/view)

Pretrained RainDiff for Shanghai Radar: [HuggingFace](https://drive.google.com/file/d/1y8BvYz3U_awm1eAYqXBy6tgbMy8t40Xr/view?usp=sharing)

### Evaluation
```shell
sh scripts/eval.sh
```
### Training
```shell
sh scripts/train.sh
```

## Acknowledgement

We refer to implementations of the following repositories and sincerely thank their contribution for the community:
- [denoising_diffusion_pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch)
- [DiffCast](https://github.com/DeminYu98/DiffCast)

## Citation
```
@misc{nguyen2025raindiffendtoendprecipitationnowcasting,
      title={RainDiff: End-to-end Precipitation Nowcasting Via Token-wise Attention Diffusion}, 
      author={Thao Nguyen and Jiaqi Ma and Fahad Shahbaz Khan and Souhaib Ben Taieb and Salman Khan},
      year={2025},
      eprint={2510.14962},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14962}, 
}
```
