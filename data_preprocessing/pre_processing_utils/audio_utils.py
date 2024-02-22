import glob
import librosa
import numpy as np
import os
import shutil
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize
import json
import pickle
import glob

def get_mfcc_features(audio, sr, num_mfcc):
    # mfcc features
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc) / 1000.
    # mfcc 1st differential
    mfcc_features_1d = mfcc_features[2:] - mfcc_features[1:-1]
    # mfcc 2nd differential
    mfcc_features_2d = mfcc_features_1d[1:] - mfcc_features_1d[:-1]
    # combine all
    mfcc_combined = np.concatenate((mfcc_features, mfcc_features_1d, mfcc_features_2d), axis=0)
    return mfcc_combined

def get_chroma_cens(audio,sr):
    chroma = librosa.feature.chroma_cens(y=audio, sr=sr).astype(np.float32)
    return chroma

def get_sc(audio,sr):
    S = np.abs(librosa.stft(audio))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    return contrast.astype(np.float32)/100.

def get_beats(audio,sr):
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    return times