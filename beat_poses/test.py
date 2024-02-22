import argparse
import os
import yaml

import numpy as np
import torch

from dataset import get_beat_pose_dataloader
from beat_pose_generator import get_beat_pose_generator
from utils.data_generator import get_random_pre_seq
from visualize import plot_beat_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beat_pose_config", type=str, default="beat_pose_config.yaml")
    parser.add_argument("--output_dir", type=str, default="test_results")
    
    return parser.parse_args()


def test(config: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    
    device = config["device"]
    
    beat_pose_generator = get_beat_pose_generator(config)
    beat_pose_generator.eval()
    
    data_path = config["dataset"]["test"]
    
    dataloader = get_beat_pose_dataloader(data_path, 1)
    
    checkpoint_path = os.path.join(config["save_dir"], "checkpoint.pt")
    
    try:
        beat_pose_generator.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        raise ValueError(f"Beat pose generator checkpoint not found: {checkpoint_path}")
    except Exception as e:
        raise ValueError(f"Error loading beat pose generator checkpoint: {e}")
    
    for idx, (_, _, mfcc, chroma, beat_idx, seq_name) in enumerate(dataloader):
        mfcc = mfcc.to(device)
        chroma = chroma.to(device)
        
        pre_seq = get_random_pre_seq()
        pre_seq = pre_seq.to(device)
        
        beat_pre_seq = torch.zeros((1, 20, 51)).to(device)
        for i, idx in enumerate(beat_idx[0]):
            if idx >= 20:
                break
            beat_pre_seq[0, i, :] = pre_seq[idx, :]
    
        with torch.no_grad():
            predicted_poses = beat_pose_generator(beat_pre_seq, mfcc, chroma)
            
        predicted_poses = plot_beat_poses(predicted_poses, f'{output_dir}/videos/{seq_name[0]}.mp4')
        predicted_poses = predicted_poses.squeeze()
        np.save(f'{output_dir}/predictions/{seq_name[0]}.npy', predicted_poses)
        
        print(f"Tested sequence {seq_name[0]}")


def main() -> None:
    args = parse_args()
    
    try:
        with open(args.beat_pose_config, "r") as f:
            beat_pose_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Beat pose config file not found: {args.beat_pose_config}")
    except Exception as e:
        raise ValueError(f"Error reading beat pose config file: {e}")
    
    test(beat_pose_config, args.output_dir)


if __name__ == "__main__":
    main()
