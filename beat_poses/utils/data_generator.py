from librosa import beat, feature
import numpy as np
import torch


def get_random_pre_seq() -> torch.Tensor:
    pre_seqs = np.load("utils/pre_seqs.pkl", allow_pickle=True)
    randint = np.random.randint(0, len(pre_seqs))
    pre_seq = pre_seqs[randint]
    pre_seq = torch.from_numpy(pre_seq)
    return pre_seq


def get_mfcc_features(music: np.ndarray) -> np.ndarray:
    mfcc_features = feature.mfcc(y=music, sr=22500, n_mfcc=14) / 1000.0
    mfcc_features_1d = mfcc_features[2:] - mfcc_features[1:-1]
    mfcc_features_2d = mfcc_features_1d[1:] - mfcc_features_1d[:-1]
    mfcc_combined = np.concatenate(
        (mfcc_features, mfcc_features_1d, mfcc_features_2d), axis=0
    )
    return mfcc_combined


def get_chroma_features(music: np.ndarray) -> np.ndarray:
    chroma = feature.chroma_cens(y=music, sr=22500).astype(np.float32)
    return chroma


def get_beat_idxs(music: np.ndarray) -> np.ndarray:
    _, beat_idxs = beat.beat_track(y=music, sr=22500)
    beat_idxs = beat_idxs[:20]
    beat_idxs = np.round(beat_idxs, 2) * 10
    beat_idxs = beat_idxs.astype(int)
    
    return beat_idxs
