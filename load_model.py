# Full model has leg loss

import datetime
import librosa
import math
import numpy as np
import random
import shutil
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from librosa.feature import mfcc
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Model_Components.m2ad_beat_generator_net import Seq2SeqGenerator
from Model_Components.m2ad_pose_repletion_net import PoseSeq2SeqGenerator, AffDiscriminator
from speech2affective_gestures.net.utils.graph import Graph
from speech2affective_gestures.net.utils.tgcn import STGraphConv, STGraphConvTranspose

class CustomGANDataset(Dataset):

    def __init__(self, device, mfcc_files, chroma_files, sc_files, beats_files, target_pose_vecs, target_labels, s2s, seq_names):
        self.device = device
        self.mfcc_files = mfcc_files
        self.chroma_files = chroma_files
        self.sc_files = sc_files
        self.beats_files = beats_files
        self.target_pose_vecs = target_pose_vecs
        self.target_labels = target_labels
        self.s2s_model = s2s

        self.seq_names = seq_names


    def __len__(self):
        return len(self.beats_files)

    def __getitem__(self, idx):

        mfcc_in = self.mfcc_files[idx]
        chroma_in = self.chroma_files[idx]
        sc_in = self.sc_files[idx]
        label_in = self.target_labels[idx].astype(np.int32) - 1
        label_in = label_in.reshape(1)

        beat_target_pose_in = self.get_beat_poses(idx)
        target_pose_in = self.target_pose_vecs[idx]

        pre_seq = self.generate_pre_seq(beat_target_pose_in, mfcc_in, chroma_in, label_in, idx)

        seq_name = self.seq_names[idx]


        return pre_seq.astype(np.float32), target_pose_in.astype(np.float32), mfcc_in.astype(np.float32), chroma_in.astype(np.float32), label_in, seq_name

    def get_beat_poses(self,idx):

        target_pose_in = self.target_pose_vecs[idx]
        beats = self.beats_files[idx]

        beat_poses = []

        for time_stamp in beats[:20]:
            frame_val = np.round(time_stamp,2) * 60
            index = int(frame_val//6)
            beat_poses.append(target_pose_in[int(index)])

        temp_frames = np.zeros((20,51))

        temp_frames[:len(beat_poses)] = np.array(beat_poses)

        return temp_frames

    def generate_pre_seq(self, beat_target_pose_in, mfcc_in, chroma_in, label_in, idx):

        beat_target_pose_in = torch.from_numpy(beat_target_pose_in).unsqueeze(0).to(device=self.device)
        mfcc_in = torch.from_numpy(mfcc_in).unsqueeze(0).to(device=self.device)
        chroma_in = torch.from_numpy(chroma_in).unsqueeze(0).to(device=self.device)
        label_in = torch.from_numpy(label_in).unsqueeze(0).to(device=self.device)

        pre_seq = beat_target_pose_in.new_zeros((beat_target_pose_in.shape[0], beat_target_pose_in.shape[1],
                                            beat_target_pose_in.shape[2] + 1))
        pre_seq[:, 0:3, :-1] = beat_target_pose_in[:, 0:3]
        pre_seq[:, 0:3, -1] = 1  

        teacher_force_ratio = 0

        self.s2s_model.eval()
        with torch.no_grad():
            outputs = self.s2s_model(pre_seq, mfcc_in, chroma_in, label_in,beat_target_pose_in, teacher_force_ratio)
        
        # How to remap outputs back to original

        outputs = outputs.squeeze(0).detach().cpu().numpy()
        beats = self.beats_files[idx]
        target_pose_in = self.target_pose_vecs[idx]


        pre_seq = np.zeros((100,51))

        pre_seq[:20] = target_pose_in[:20]
        

        beat_counter = 0
        for time_stamp in beats[:20]:
            
            frame_val = np.round(time_stamp,2) * 60
            index = int(frame_val//6)

            if index >20:
                pre_seq[index] = outputs[beat_counter]
                
            beat_counter+=1

        constraint_bit = np.ones((100,1))

        pre_seq = np.hstack((pre_seq, constraint_bit))

        return pre_seq

def generate_output(s2ag_generator, test_sample):

    pre_seq, target_poses, mfcc_in, chroma_in, label_in, seq_name = test_sample

    # pre_seq = torch.from_numpy(pre_seq).unsqueeze(0).to(device=device)
    # target_poses = torch.from_numpy(target_poses).unsqueeze(0).to(device=device)
    # mfcc_in = torch.from_numpy(mfcc_in).unsqueeze(0).to(device=device)
    # chroma_in = torch.from_numpy(chroma_in).unsqueeze(0).to(device=device)
    # label_in = np.array([1])
    # label_in = torch.from_numpy(label_in).unsqueeze(0).to(device=device)
    device = 'cuda'
    pre_seq = pre_seq.to(device=device)
    target_poses = target_poses.to(device=device)
    mfcc_in = mfcc_in.to(device=device)
    chroma_in = chroma_in.to(device=device)
    label_in = np.array([1])
    label_in = torch.from_numpy(label_in).unsqueeze(0).to(device=device)
    
    with torch.no_grad():
        out_dir_vec = s2ag_generator(pre_seq,mfcc_in,chroma_in,label_in, target_poses, 0)

    return out_dir_vec

dir_vec_pairs = [
    (0,1,10), (0,2,14), (0,8,14), (0,5,30), (0,11,30),
    (5,6,20), (6,7,20), (11,12,20), (12,13,20),
    (2,3,16), (3,4,14), (8,9,16), (9,10,14),
    (1,14,4), (14,16,4), (1,15,4), (15,17,4)
]

dir_edge_pairs = [
    (0,1), (0,2), (0,3), (0,4), (0,13), (0,15),
    (3,5), (5,6), (4,7), (7,8), (1,9), (9,10),
    (2,11), (11,12), (13,14), (15,16)
]

body_parts_edge_idx = [

    [13,14,15,16], #head
    [0,3,4], #torso
    [1,9,10], #left arm
    [2,11,12], #right arm
    [5,6], #left leg
    [7,8] #right leg
]

max_body_part_edges = 4
body_parts_edge_pairs = [
    (1,0), (1,2), (1,3), 
    (1,4), (1,5)
]
def convert_dir_vec_to_pose(vec):
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((18, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]

    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 18, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def obtain_joints(output_vecs, save_name):

    final_out = output_vecs.squeeze(0).detach().cpu().numpy()
    print(final_out.shape)
    joints = convert_dir_vec_to_pose(final_out)
    with open(save_name,'wb') as f:
        np.save(f,joints)

    return joints




def main():

    print("Script started...")


    encoder_args = {
        'hidden_size':256,
        'n_layers':3,
        'dropout_prob':0.4
    }

    decoder_args = {
        'hidden_size':256,
        'n_layers':3,
        'dropout_prob':0.4
    }

    device = 'cuda'

    s2ag_generator = PoseSeq2SeqGenerator(encoder_args, decoder_args, 440, 37, 440, 12, 2, 100).to(device=device)
    s2ag_discriminator = AffDiscriminator(2, 100, 16).to(device=device)

    s2s = Seq2SeqGenerator(encoder_args, decoder_args, 440, 37, 440, 12, 2, 20).to(device=device)
    s2s.load_state_dict(torch.load('./s2s_AIST.pth'))
    for params in s2s.parameters():
        params.requires_grad = False
    s2s.eval()

    s2ag_generator.load_state_dict(torch.load('./BN_RN_TFModif_v5/M2AD_Gen11_ep250_10FPS'))
    s2ag_discriminator.load_state_dict(torch.load('./BN_RN_TFModif_v5/M2AD_Disc11_ep350_10FPS'))

    s2ag_generator.eval()

    with open('./AIST_Test_data_with_root.npy','rb') as f:
        mfcc_files = np.load(f)
        chroma_files = np.load(f)
        sc_files = np.load(f)
        beats_files = np.load(f,allow_pickle=True)
        seq_names = np.load(f)
        target_pose_vecs = np.load(f)
        target_pose_vertices = np.load(f)
        target_labels = np.load(f)
        root_properties_data = np.load(f)

    gan_data = CustomGANDataset(device, mfcc_files,chroma_files,sc_files,beats_files,target_pose_vecs,target_labels, s2s.to(device=device), seq_names)

    test_dataloader = DataLoader(gan_data, batch_size=1, shuffle=True)    

    for idx, test_sample in enumerate(test_dataloader):
        out_vectors = generate_output(s2ag_generator,test_sample)
        save_name = './BN_RN_Outputs_v5/'+str(idx)+'.npy'
        joints = obtain_joints(out_vectors, save_name)


# In[10]:
if __name__ == "__main__":
    main()