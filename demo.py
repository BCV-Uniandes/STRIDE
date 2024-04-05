# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import json
import random
import time
from pathlib import Path
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import to_device
import util.misc as utils

from datasets import build_dataset
import warnings
warnings.filterwarnings("ignore")

# breakpoint()
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', type=str, default="config/STRIDE/STRIDE_4scale.py")
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--azure_download', action='store_true', help='Download images from azure')
    parser.add_argument('--connection_string', type=str, help='Connection string to azure blob storage')
    parser.add_argument('--container_name', type=str, help='Name of he container with the images')
    parser.add_argument('--azure_list_path', type=str, help='Path to txt with list of images to download')
    parser.add_argument('--demo_images_path', type=str, help='Path to downloaded images')
    parser.add_argument('--pretrain_model', type=str, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    
    # Distributed params
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    return parser

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def demo(args):

    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    cfg_dict['regression'] = False
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    # build model
    model, _, postprocessors = build_func(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    if args.azure_download:
        images_list = open(args.azure_list_path,'r').readlines()
    else:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        images_list = sorted(os.listdir(args.demo_images_path))
        images_list = [file for file in images_list if os.path.splitext(file)[1].lower() in image_extensions]
        # 'categories': [{'name': str(i), 'id': i+1, 'supercategory': 'demo_category'} for i in range(27)]
        images_json = {'categories': [{'name': str(i), 'id': i+1, 'supercategory': 'demo_category'} for i in range(27)], 'images': [], 'annotations': []}
        for im_id, img in enumerate(images_list):
            images_json['images'].append({'id': im_id, 
                                        'file_name': img,
                                        'width': 13312,
                                        'height': 4000})
        
        with open(os.path.join(args.output_dir,'temp_json.json'), 'w') as f:
            json.dump(images_json, f)

        
    if os.path.isdir(os.path.join(args.output_dir,'Inferences')):
        infered_images = set(im.replace('.pth','.jpg') for im in os.listdir(os.path.join(args.output_dir,'Inferences')))
        images_list = [im.strip() for im in images_list if im.strip() not in infered_images]
    else:
        os.makedirs(os.path.join(args.output_dir,'Inferences'), exist_ok=True)
        
    dataset_demo = build_dataset(image_set='demo', args=args)

    if args.distributed:
        sampler_demo = DistributedSampler(dataset_demo, shuffle=False)
    else:
        sampler_demo = torch.utils.data.SequentialSampler(dataset_demo)

    data_loader_demo = DataLoader(dataset_demo, args.batch_size, sampler=sampler_demo,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.pretrain_model, map_location='cpu')['model']
    from collections import OrderedDict
    _ignorekeywordlist = []
    ignorelist = []

    def check_keep(keyname, ignorekeywordlist):
        for keyword in ignorekeywordlist:
            if keyword in keyname:
                ignorelist.append(keyname)
                return False
        return True

    logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
    _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

    _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
    logger.info(str(_load_output))

    with torch.no_grad():
        model.eval()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Inference:'

        # for debug only
        for samples, targets in metric_logger.log_every(data_loader_demo, 10, header, logger=logger):
            samples = samples.to(device)

            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            for target_info, prediction in zip(targets, results):
                coco_results = []
                original_id = -1 
                original_id = target_info['image_id'].item()
                if len(prediction) == 0 or original_id==-1:
                    continue
                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()
                if not isinstance(prediction["scores"], list):
                    scores = prediction["scores"].tolist()
                else:
                    scores = prediction["scores"]
                if not isinstance(prediction["labels"], list):
                    labels = prediction["labels"].tolist()
                else:
                    labels = prediction["labels"]

                coco_results = [
                                {
                                    "image_id": original_id,
                                    "category_id": labels[k],
                                    "bbox": box,
                                    "score": scores[k],
                                }
                                for k, box in enumerate(boxes)
                            ]
            
                if len(coco_results):
                    image_name = images_list[original_id].strip()
                    torch.save(coco_results,os.path.join(args.output_dir, 'Inferences', image_name.replace('.jpg','.pth')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.demo = True
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    demo(args)