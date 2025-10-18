
import torch
import os
import numpy as np
import os.path as osp
import datetime

from functools import partial
from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def save_colorbar(color_map, bounds, orientation='horizontal', invert=False, filename='colorbar.png'):
    cmap = colors.ListedColormap(color_map)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Figure size depending on orientation
    if orientation == 'horizontal':
        fig, ax = plt.subplots(figsize=(6, 0.5))
    else:
        fig, ax = plt.subplots(figsize=(0.25, 6))

    fig.subplots_adjust(bottom=0.5 if orientation == 'horizontal' else 0.05)

    tick_positions = bounds[::2]

    cbar = plt.colorbar(
        sm, cax=ax, orientation=orientation,
        boundaries=bounds, ticks=tick_positions,
        drawedges=False  # No black lines between colors
    )

    # Flip for vertical if needed
    if orientation == 'vertical' and invert:
        cbar.ax.invert_yaxis()

    # Remove outer border
    cbar.outline.set_visible(False)

    # Remove tick marks but keep numbers
    cbar.ax.tick_params(which='both', length=0)

    # Save to file
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

def vis_res(pred_seq, gt_seq, in_seq, save_path, data_type='vil',
            save_grays=False, do_hmf=False, save_colored=True,save_gif=False,
            pixel_scale = None, thresholds = None, gray2color = None
            ):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor) or isinstance(gt_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
        in_seq = in_seq.detach().cpu().numpy()
        # pred_seq_ = pred_seq_.detach().cpu().numpy()

    pred_seq = pred_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    in_seq = in_seq.squeeze()
    # pred_seq_ = pred_seq_.squeeze()

    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(osp.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):            
            plt.imsave(osp.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmax=1.0, vmin=0.0)
            plt.imsave(osp.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmax=1.0, vmin=0.0)


    if data_type=='vil':
        pred_seq = pred_seq * pixel_scale
        pred_seq = pred_seq.astype(np.uint8)

        gt_seq = gt_seq * pixel_scale
        gt_seq = gt_seq.astype(np.uint8)

        in_seq = in_seq * pixel_scale
        in_seq = in_seq.astype(np.uint8)
    
    colored_pred = np.array([gray2color(pred_seq[i], data_type=data_type) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt =  np.array([gray2color(gt_seq[i], data_type=data_type) for i in range(len(gt_seq))],dtype=np.float64)
    colored_ip = np.array([gray2color(in_seq[i], data_type=data_type) for i in range(len(in_seq))],dtype=np.float64)

    if save_colored:
        os.makedirs(osp.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'input_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'baseline'), exist_ok=True)

        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(osp.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(osp.join(save_path, 'targets_colored', f'{i}.png'), gt)

        for i, input in enumerate(colored_ip):
            plt.imsave(osp.join(save_path, 'input_colored', f'{i}.png'), input)


    grid_pred = np.concatenate([
        np.concatenate([i for i in colored_pred], axis=-2),
    ], axis=-3)

    grid_gt = np.concatenate([
        np.concatenate([i for i in colored_gt], axis=-2,),
    ], axis=-3)
    
    print(grid_pred.shape, grid_gt.shape)
    # grid_concat = np.concatenate([grid_pred_, grid_pred, grid_gt], axis=-3,)
    grid_concat = np.concatenate([grid_pred, grid_gt], axis=-3,)
    plt.imsave(osp.join(save_path, 'all.png'), grid_concat)
    
    if save_gif:
        clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'pred.gif'), fps=4, verbose=False)
        clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'targets.gif'), fps=4, verbose=False)
    
    if do_hmf:
        def hit_miss_fa(y_true, y_pred, thres):
            mask = np.zeros_like(y_true)
            mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
            mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
            mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
            mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
            return mask
            
        grid_pred = np.concatenate([
            np.concatenate([i for i in pred_seq], axis=-1),
        ], axis=-2)
        grid_gt = np.concatenate([
            np.concatenate([i for i in gt_seq], axis=-1),
        ], axis=-2)

        hmf_mask = hit_miss_fa(grid_pred, grid_gt, thres=thresholds[2])
        plt.axis('off')
        plt.imsave(osp.join(save_path, 'hmf.png'), hmf_mask, cmap=colors.ListedColormap(HMF_COLORS))


DATAPATH = {
    'cikm' : './resources/data/cikm_2.5km.h5',
    'shanghai' : './resources/data/shanghai.h5',
    'meteo' : './resources/data/meteo_NW.h5',
    'sevir' : './resources/data/sevir/',
    'your_dataset_name': './resources/data/your_dataset_name/', #'edit here'
}

def get_dataset(data_name, img_size, seq_len, **kwargs):
    dataset_name = data_name.lower()
    train = val = test = None

    if dataset_name == "your_dataset_name":
        """
        'edit here'
        Define your dataset class dataset_your_dataset_name in /RainDiff/datasets (inherits from torch.utils.data.Dataset), where target shape of train[0]: [seq_len, 1, img_size, img_size], where:
            seq_len = frames_in + frames_out
            H, W    = img_size, img_size (image dimensions)
        """
        from .dataset_your_dataset_name import dust, gray2color, THRESHOLDS, PIXEL_SCALE
        train = dataset_your_dataset_name(DATAPATH[data_name], img_size, 'train', )
        val = dataset_your_dataset_name(DATAPATH[data_name], img_size, 'val')
        test = dataset_your_dataset_name(DATAPATH[data_name],img_size, 'test')

        pass
        
    elif dataset_name == 'cikm':
        from .dataset_cikm import CIKM, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train = CIKM(DATAPATH[data_name], img_size, 'train', )
        # val = CIKM(DATAPATH[data_name], 'valid', img_size)
        val = CIKM(DATAPATH[data_name], img_size, 'val')
        test = CIKM(DATAPATH[data_name],img_size, 'test')
        
    elif data_name == 'shanghai':
        from .dataset_shanghai import Shanghai, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Shanghai(DATAPATH[data_name], type='train', img_size=img_size)
        val = Shanghai(DATAPATH[data_name], type='val', img_size=img_size)
        test = Shanghai(DATAPATH[data_name], type='test', img_size=img_size)
    
    elif data_name == 'meteo':
        from .dataset_meteonet import Meteo, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Meteo(DATAPATH[data_name], type='train', img_size=img_size)
        val = Meteo(DATAPATH[data_name], type='val', img_size=img_size)
        test = Meteo(DATAPATH[data_name], type='test', img_size=img_size)
        
    elif dataset_name == 'sevir':
        from .dataset_sevir import SEVIRTorchDataset, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train_valid_split = (2019, 1, 1)
        valid_test_split = (2019, 6, 1)#(2019, 6, 1)
        test_end_date = (2019, 12, 31)
        batch_size = kwargs.get('batch_size', 1)
        
        train = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            traing='train',
            img_size=img_size,
            shuffle=True,
            seq_len=25,
            stride=5,      # ?
            sample_mode='sequent',
            batch_size=batch_size,
            num_shard=1,
            rank=0,
            start_date=None, # datetime.datetime(*(2018, 6, 1)), 
            end_date=datetime.datetime(*train_valid_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        val = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            traing='val',
            img_size=img_size,
            shuffle=False,
            seq_len=25,
            stride=5,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*train_valid_split),
            end_date=datetime.datetime(*valid_test_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        test = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            traing='test',
            shuffle=False,
            img_size=img_size,
            seq_len=25,
            stride=5,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*valid_test_split),
            end_date=None,
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not available")


    color_fn = partial(vis_res, 
                    pixel_scale = PIXEL_SCALE, 
                    thresholds = THRESHOLDS, 
                    gray2color = gray2color)
    
    return train, val, test, color_fn, PIXEL_SCALE, THRESHOLDS
