import argparse
import os
import numpy as np
from PIL import Image

import torch
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  default='', help='模型配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='', help='模型检查点文件路径')
    parser.add_argument('--image', type=str, default='', help='输入图像路径')
    parser.add_argument('--output', type=str, default='', help='输出图像路径')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def load_model(config_file, checkpoint_file, device):
    register_all_modules()
    model = init_detector(config_file, checkpoint_file, device=device)
    return model


def process_image(model, image_path, output_path):

    image = mmcv.imread(image_path, channel_order='rgb')
    result = inference_detector(model, image)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=None,
        wait_time=0,
        pred_score_thr=0.85
    )
    res_img = visualizer.get_image()
    
    arr_uint8 = res_img.astype('uint8')
    img = Image.fromarray(arr_uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"结果已保存至: {output_path}")


def main():

    args = parse_args()
    model = load_model(args.config, args.checkpoint, args.device)
    process_image(model, args.image, args.output)


if __name__ == "__main__":
    main()