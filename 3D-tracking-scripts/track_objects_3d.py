import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import time
from dataclasses import dataclass
from torchvision.ops import nms
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from tqdm import tqdm
import argparse
import pickle
import os
import sys
import depth_pro
import matplotlib
import pdb
import torch.nn.functional as F
import cv2
from PIL import Image
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

    
def add_timestep_index(example, index):
    
    def add_step_index(example, index):
        example['timestep'] = index
        return example
    
    timestep_index = tf.range(index['timestep_length'])
    timestep_index = tf.data.Dataset.from_tensor_slices(timestep_index)
    example['steps'] = tf.data.Dataset.zip((example['steps'], timestep_index))
    example['steps'] = example['steps'].map(add_step_index)
    example['idx'] = index['idx']
    
    return example


def config():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-shard', type=int, default=0,
                        help='Shard of the dataset to save', choices=[i for i in range(1024)])
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--pickle_file_path', type=str, default='segment_images.pkl')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--checkpoint-path', type=str, default='/home/ubuntu/projects/akshay/Depth_Anything_V2/checkpoints')
    args = parser.parse_args()
    return args

def get_final_depth_map(canonical_inverse_depth, f_px, fov_deg, resize=True):
    
    H, W = 256, 256
    if f_px is None:
        f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        
    inverse_depth = canonical_inverse_depth * (W / f_px)
    f_px = f_px.squeeze()

    if resize:
        inverse_depth = nn.functional.interpolate(
            inverse_depth.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        )

    depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
    
    # Normalize depth map and convert to numpy
    depth = depth.squeeze().detach().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    return depth


if __name__ == '__main__':
    
    params = config()
    
    shard = params.data_shard
    split = f'train[{shard}shard]'
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    dataset = tfds.load('fractal20220817_tracking_data', data_dir=params.data_dir,
                        split=split)
    
    data_dict = {'idx': [idx for idx in range(len(dataset))],
                 'timestep_length': [len(item['steps']) for item in dataset]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = tf.data.Dataset.zip((dataset, data_idx))
    dataset = dataset.map(add_timestep_index, num_parallel_calls=1)
    
    # Load depth pro model
    model, transform = depth_pro.create_model_and_transforms()
    model.to(device)
    model.eval()
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    
    print(f'Starting to segment key objects for shard {shard}...')
    start_time = time.time()
    
    img_idx = 0
    images_data = {}

    # for example in dataset:
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        
        task = [d['observation']['natural_language_instruction'].numpy().decode('utf-8') for d in example['steps'].take(1)][0]
        example_idx = int(example['idx'].numpy())
        depth_map_list = []
        
        for b_idx, batch in tqdm(enumerate(example['steps'].batch(params.batch_size)), total=len(example['steps'])//params.batch_size):
            
            # Load transform/resize the images
            orig_images = batch['observation']['image']
            images = torch.stack([transform(depth_pro.load_rgb(Image.fromarray(i.numpy()))[0]) for i in orig_images])
            f_px_list = [depth_pro.load_rgb(Image.fromarray(i.numpy()))[1] for i in orig_images]
            images = nn.functional.interpolate(images, size=(model.img_size, model.img_size), mode='bilinear', align_corners=False)
            images = images.to(device)

            
            N = len(batch['observation']['natural_language_instruction'])
            ts_list = [int(b.numpy()) for b in batch['timestep']]
            
            # Load end effector locations, object locations and object distances
            end_effector_locs = [batch['observation']['end_effector_loc'][i].numpy().tolist()
                                 for i in range(N)]
            object_locations = [batch['observation']['object_locs'][i].numpy().tolist()
                                for i in range(N)]
            object_distances = [batch['observation']['object_distances'][i].numpy().tolist()
                                for i in range(N)]
            
            with torch.no_grad():
                canonical_inverse_depths, fov_degs = model.forward(images)
            
            depth_map_list = []
            for canonical_inverse_depth, f_px, fov_deg in zip(canonical_inverse_depths, f_px_list, fov_degs):
                depth_map = get_final_depth_map(canonical_inverse_depth, f_px, fov_deg)
                depth_map_list.append(depth_map)
            
            for depth, ts, ee_loc, object_loc_list, object_dist_list in zip(depth_map_list, ts_list, end_effector_locs, object_locations, object_distances):
                
                # Get the integer coordinates of the end effector location
                img_ee_loc = [round(ee_loc[0]*256), round(ee_loc[1]*256)]
                
                # Get depth at end effector location
                normalized_ee_point = depth[img_ee_loc[1], img_ee_loc[0]]
                
                object_3d_locations = []
                object_3d_distances = []
                for i in range(4):
                    
                    object_loc = object_loc_list[2*i:2*i+2]
                    
                    if object_loc != [-2.0, -2.0]:
                        object_loc = [round(object_loc[0]*256), round(object_loc[1]*256)]
                        object_3d_locations.append(depth[object_loc[1], object_loc[0]])
                        object_3d_distances.append(normalized_ee_point - depth[object_loc[1], object_loc[0]])
                    else:
                        object_3d_locations.append(-2.0)
                        object_3d_distances.append(-2.0)
                    
                new_object_loc_list, new_object_dist_list = [], []
                new_ee_loc = ee_loc + [normalized_ee_point]
                
                for i in range(4):
                    new_object_loc_list.extend(object_loc_list[2*i:2*i+2] + [object_3d_locations[i]])
                    new_object_dist_list.extend(object_dist_list[2*i:2*i+2] + [object_3d_distances[i]])
                
                # print(f"New object locations: {new_object_loc_list}")
                # print(f"New object distances: {new_object_dist_list}")
                # print(f"New end effector location: {new_ee_loc}")
                
                img_name = f"{task}_{example_idx}_{ts}.png"
                images_data[img_name] = {
                    'end effector image location': new_ee_loc,
                    'object locations': new_object_loc_list,
                    'object distances': new_object_dist_list
                }
                img_idx += 1

           
    print(f'Saving {img_idx} 3D-tracks to pickle file...')
    pickle_file = params.pickle_file_path
    with open(pickle_file, 'wb') as f:
        pickle.dump(images_data, f)
    print(f"Time taken: {time.time() - start_time}")
