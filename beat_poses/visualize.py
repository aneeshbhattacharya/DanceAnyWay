import os

import cv2
import numpy as np
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
from moviepy.editor import AudioFileClip
import torch
from torch import Tensor

from utils.draw import Plotter3d


def convert_dir_vec_to_pose(vec: Tensor) -> np.ndarray:
    dir_vec_pairs = [
        (0, 1, 10),
        (0, 2, 14),
        (0, 8, 14),
        (0, 5, 30),
        (0, 11, 30),
        (5, 6, 20),
        (6, 7, 20),
        (11, 12, 20),
        (12, 13, 20),
        (2, 3, 16),
        (3, 4, 14),
        (8, 9, 16),
        (9, 10, 14),
        (1, 14, 4),
        (14, 16, 4),
        (1, 15, 4),
        (15, 17, 4),
    ]
    
    batch_size, seq_len, _ = vec.shape
    device = vec.device
    
    vec = vec.reshape(batch_size, seq_len, 17, 3)
    joint_pos = torch.zeros((vec.shape[0], vec.shape[1], 18, 3)).to(device)
    
    for j, pair in enumerate(dir_vec_pairs):
        joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]

    joint_pos = joint_pos.cpu().numpy()
    return joint_pos

    
def get_hip_positions(l_hip: np.ndarray, r_hip: np.ndarray, root_joint: np.ndarray) -> np.ndarray:
    dir_vec = r_hip - l_hip
    vec = dir_vec / np.sqrt(np.sum(np.multiply(dir_vec, dir_vec)))
    new_l_hip = root_joint - 5 * vec
    new_r_hip = root_joint + 5 * vec

    return new_l_hip, new_r_hip


def get_root_position(neck: np.ndarray, root_joint: np.ndarray) -> np.ndarray:
    dir_vec = neck - root_joint
    vec = dir_vec / np.sqrt(np.sum(np.multiply(dir_vec, dir_vec)))
    new_root = root_joint + 3 * vec
    
    return new_root


def repeat_poses(poses: np.ndarray, num_frames: int) -> np.ndarray:
    new_poses = []
    
    for pose in poses:
        for _ in range(num_frames):
            new_poses.append(pose)
            
    return np.array(new_poses)


def stitch_video(frames_dir: str, output_path: str) -> None:
    frame_files = os.listdir(frames_dir)
    frame_files.sort()
    frame_files = [os.path.join(frames_dir, frame_file) for frame_file in frame_files]
    
    video = ImageSequenceClip.ImageSequenceClip(frame_files, fps=30)
        
    video.write_videofile(output_path)
    

def plot_poses(poses: np.ndarray, output_path: str) -> None:
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    
    os.makedirs('temp_poses', exist_ok=True)

    for i, pose in enumerate(poses):
        stuff_val = np.array([0.5, 0.5, 0.5])
        
        l_hip = pose[5]
        r_hip = pose[11]
        neck = pose[0]
        
        root = np.vstack((l_hip, r_hip)).mean(axis=0)
        root = get_root_position(neck, root)
        
        part1 = pose[:2]
        part2 = pose[2:]
        
        temp_pose = np.vstack((part1, stuff_val))
        pose = np.vstack((temp_pose, part2))
        l_hip, r_hip = get_hip_positions(l_hip, r_hip, root)
        
        pose[6] = l_hip
        pose[12] = r_hip
        pose[2] = root
        
        pose[:, 2] = pose[:, 2] + 120
        
        pose = pose.reshape(1, 19, 3)
        
        edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(pose.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        
        plotter.plot(canvas_3d, pose, edges)
        
        cv2.imwrite(f'temp_poses/{i}.png', canvas_3d)
    
    stitch_video('temp_poses', output_path)
    
    os.removedirs('temp_poses')


def plot_beat_poses(dir_vecs: Tensor, output_path: str, num_frames: int = 10) -> np.ndarray:
    poses = convert_dir_vec_to_pose(dir_vecs)
    
    repeated_poses = repeat_poses(poses, num_frames)
    repeated_poses = repeated_poses.reshape(-1, 18, 3)
    
    plot_poses(repeated_poses, output_path)
    
    return poses
    