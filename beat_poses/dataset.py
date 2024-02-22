import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def collate_fn(batch: List) -> Tuple:
    pre_seq, target, mfcc, chroma, beat_idx, seq_name = zip(*batch)

    pre_seq = torch.from_numpy(np.array(pre_seq)).float()
    target = torch.from_numpy(np.array(target)).float()
    mfcc = torch.from_numpy(np.array(mfcc)).float()
    chroma = torch.from_numpy(np.array(chroma)).float()
    beat_idx = list(beat_idx)
    
    return pre_seq, target, mfcc, chroma, beat_idx, seq_name


class BeatPoseDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        self.mfccs = data["audio_mfccs"]
        self.chromas = data["audio_chroma"]
        self.beat_idxs = data["audio_beats"]
        self.targets = data["target_pose_vecs"]
        self.seq_names = data["seq_name"]
        
        for i in range(len(self.beat_idxs)):
            self.beat_idxs[i] = np.array(self.beat_idxs[i])
            self.beat_idxs[i] = np.round(self.beat_idxs[i], 2) * 60
            self.beat_idxs[i] = self.beat_idxs[i] // 6
            self.beat_idxs[i] = self.beat_idxs[i].astype(int)
            
        self.seq_names = [name.split("/")[-1].split(".")[0] for name in self.seq_names]

    def __len__(self) -> int:
        return len(self.seq_names)
    
    def __getitem__(self, idx) -> tuple:
        pre_seq = self.generate_pre_seq(idx)
        mfcc = self.mfccs[idx]
        chroma = self.chromas[idx]
        beat_idx = self.beat_idxs[idx][:20]
        target = self.generate_target(idx)
        seq_name = self.seq_names[idx]

        return (pre_seq, target, mfcc, chroma, beat_idx, seq_name)
    
    def generate_pre_seq(self, idx: int) -> np.ndarray:
        beat_idxs = self.beat_idxs[idx]
        beat_idxs = beat_idxs[beat_idxs < 20]
        
        pre_seq = np.zeros((20, 51))
        pre_seq[: len(beat_idxs)] = self.targets[idx][beat_idxs]

        return pre_seq

    def generate_target(self, idx: int) -> np.ndarray:
        beat_idxs = self.beat_idxs[idx]
        beat_idxs = beat_idxs[:20]

        target = np.zeros((20, 51))
        target[: len(beat_idxs)] = self.targets[idx][beat_idxs]

        return target


def get_beat_pose_dataloader(data_path: str, batch_size: int) -> DataLoader:
    dataset = BeatPoseDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader
