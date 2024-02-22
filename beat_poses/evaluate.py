import argparse
import yaml
from typing import Tuple

import librosa
import numpy as np
import torch
from torch import Tensor

from beat_pose_generator import get_beat_pose_generator
from utils.data_generator import (
    get_random_pre_seq,
    get_mfcc_features,
    get_chroma_features,
    get_beat_idxs,
)
from visualize import plot_beat_poses

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beat_pose_config", type=str, default="beat_pose_config.yaml")
    parser.add_argument("--beat_pose_checkpoint", type=str, default="train_results/beat_pose_generators/checkpoint.pt")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--music_file", type=str, required=True)

    return parser.parse_args()


def get_music_clip(music_file:str, duration:int) -> np.ndarray:
    music = librosa.load(music_file, sr=22500)[0]
    music = music[:int(duration * 22500)]
    return music


def get_music_features(music: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mfcc = get_mfcc_features(music)
    chroma = get_chroma_features(music)
    beat_idxs = get_beat_idxs(music)
    
    mfcc = torch.tensor(mfcc).float().unsqueeze(0)
    chroma = torch.tensor(chroma).float().unsqueeze(0)
    
    return mfcc, chroma, beat_idxs


def get_beat_pre_seq(pre_seq: Tensor, beat_idxs: np.ndarray) -> Tensor:
    beat_pre_seq = torch.zeros(1, 20, 51)
    for i, idx in enumerate(beat_idxs):
        if idx >= 20:
            break
        beat_pre_seq[0, i, :] = pre_seq[0, idx, :]

    return beat_pre_seq


def assign_beat_poses(pre_seq: Tensor, beat_poses: Tensor, beat_idxs: np.ndarray) -> Tensor:
    zeros = torch.zeros((1, 50, 51))
    pre_seq = torch.cat((pre_seq, zeros), dim=1)
    
    original_length = beat_idxs.shape[0]
    beat_idxs = beat_idxs[beat_idxs > 20]
    new_length = beat_idxs.shape[0]
    start_idx = original_length - new_length
    pre_seq[0, beat_idxs, :] = beat_poses[0, start_idx:, :]
    
    return pre_seq


def beat_pose_eval(beat_pose_generator: torch.nn.Module,
                   pre_seq: Tensor,
                   mfcc: Tensor,
                   chroma: Tensor,
                   beat_idxs: np.ndarray) -> Tensor:
    beat_pre_seq = get_beat_pre_seq(pre_seq, beat_idxs)
    with torch.no_grad():
        beat_poses = beat_pose_generator(beat_pre_seq, mfcc, chroma)
        
    return beat_poses


def main():
    args = parse_args()
    
    try:
        with open(args.beat_pose_config, "r") as f:
            beat_pose_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Beat pose config file not found: {args.beat_pose_config}")
    except Exception as e:
        raise ValueError(f"Error reading beat pose config file: {e}")
    
    beat_pose_generator = get_beat_pose_generator(beat_pose_config)
    beat_pose_generator.eval()
    
    try:
        beat_pose_generator.load_state_dict(torch.load(args.beat_pose_checkpoint))
    except FileNotFoundError:
        raise ValueError(f"Beat pose generator checkpoint not found: {args.beat_pose_checkpoint}")
    except Exception as e:
        raise ValueError(f"Error loading beat pose generator checkpoint: {e}")

    music = get_music_clip(args.music_file, beat_pose_config["duration"])
    mfcc, chroma, beat_idxs = get_music_features(music)
    pre_seq = get_random_pre_seq()
    
    beat_poses = beat_pose_eval(beat_pose_generator, pre_seq, mfcc, chroma, beat_idxs)
    beat_poses = plot_beat_poses(beat_poses, f'{args.output}.mp4', music=args.music_file)
    beat_poses = beat_poses.squeeze()
    np.save(f'{args.output}.npy', beat_poses)
    
    print(f"Output saved to {args.output}.npy, {args.output}.mp4")


if __name__ == "__main__":
    main()
