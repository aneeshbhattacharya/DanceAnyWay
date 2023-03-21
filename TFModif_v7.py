# Old model with 0.5 TF ration and has leg loss

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
from torch.utils.data import Dataset, DataLoader


RIGHT_FEMUR_BONE_IDX = 5
RIGHT_SHIN_BONE_IDX = 6
LEFT_FEMUR_BONE_IDX = 7
LEFT_SHIN_BONE_IDX = 8

RIGHT_FOOT_JOINT_IDX = 15
LEFT_FOOT_JOINT_IDX = 7

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

in_shape = torch.rand(8,20,51)

s2s = Seq2SeqGenerator(encoder_args, decoder_args, 440, 37, 440, 12, 2, 20).to(device=device)
s2s.load_state_dict(torch.load('./s2s_AIST.pth'))
for params in s2s.parameters():
    params.requires_grad = False
s2s.eval()



with open('./AIST_Train_data_with_root.npy','rb') as f:
    mfcc_files = np.load(f)
    chroma_files = np.load(f)
    sc_files = np.load(f)
    beats_files = np.load(f,allow_pickle=True)
    seq_names = np.load(f)
    target_pose_vecs = np.load(f)
    target_pose_vertices = np.load(f)
    target_labels = np.load(f)
    root_properties_data = np.load(f)
    
class CustomGANDataset(Dataset):

    def __init__(self,mfcc_files,chroma_files,sc_files,beats_files,target_pose_vecs,target_labels, s2s, seq_names):
        self.mfcc_files = mfcc_files
        self.chroma_files = chroma_files
        self.sc_files = sc_files
        self.beats_files = beats_files
        self.target_pose_vecs = target_pose_vecs
        self.target_labels = target_labels
        self.s2s_model = s2s

        self.seq_names = seq_names


    def __len__(self):
        return len(beats_files)

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


        return pre_seq, target_pose_in, mfcc_in, chroma_in, label_in, seq_name

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

        beat_target_pose_in = torch.from_numpy(beat_target_pose_in).unsqueeze(0).to(device=device)
        mfcc_in = torch.from_numpy(mfcc_in).unsqueeze(0).to(device=device)
        chroma_in = torch.from_numpy(chroma_in).unsqueeze(0).to(device=device)
        label_in = torch.from_numpy(label_in).unsqueeze(0).to(device=device)

        pre_seq = beat_target_pose_in.new_zeros((beat_target_pose_in.shape[0], beat_target_pose_in.shape[1],
                                            beat_target_pose_in.shape[2] + 1))
        pre_seq[:, 0:3, :-1] = beat_target_pose_in[:, 0:3]
        pre_seq[:, 0:3, -1] = 1  

        teacher_force_ratio = 0

        self.s2s_model.eval()
        with torch.no_grad():
            outputs = self.s2s_model(pre_seq, mfcc_in\
                            , chroma_in, label_in,\
                            beat_target_pose_in, teacher_force_ratio)
        
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

gan_data = CustomGANDataset(mfcc_files,chroma_files,sc_files,beats_files,target_pose_vecs,target_labels, s2s.to(device=device), seq_names)

train_dataloader = DataLoader(gan_data, batch_size=8, shuffle=True)    


class ReconstructSkeleton():

    def __init__(self, device):
        
        self.dir_vec_pairs = [
            (0, 1, 10), (0, 2, 14), (0, 8, 14), (0, 5, 30), (0, 11, 30),
            (5, 6, 20), (6, 7, 20), (11, 12, 20), (12, 13, 20),
            (2, 3, 16), (3, 4, 14), (8, 9, 16), (9, 10, 14),
            (1, 14, 4), (14, 16, 4), (1, 15, 4), (15, 17, 4)
        ]

        self.dir_edge_pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 13), (0, 15),
            (3, 5), (5, 6), (4, 7), (7, 8), (1, 9), (9, 10),
            (2, 11), (11, 12), (13, 14), (15, 16)
        ]

        self.body_parts_edge_idx = [

            [13, 14, 15, 16],  #head
            [0, 3, 4],  #torso
            [1, 9, 10],  #left arm
            [2, 11, 12],  #right arm
            [5, 6],  #left leg
            [7, 8]  #right leg
        ]

        self.max_body_part_edges = 4
        self.body_parts_edge_pairs = [
            (1, 0), (1, 2), (1, 3), 
            (1, 4), (1, 5)
        ]

        self.device = device

    def convert_dir_vec_to_pose(self, vec):
        # vec = np.array(vec)

        if vec.shape[-1] != 3:
            vec = vec.view(vec.shape[:-1] + (-1, 3))

        if len(vec.shape) == 2:
            joint_pos = torch.zeros((18, 3)).to(self.device)
            for j, pair in enumerate(self.dir_vec_pairs):
                joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]

        elif len(vec.shape) == 3:
            joint_pos = torch.zeros((vec.shape[0], 18, 3)).to(self.device)
            for j, pair in enumerate(self.dir_vec_pairs):
                joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
        elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
            joint_pos = torch.zeros((vec.shape[0], vec.shape[1], 18, 3)).to(self.device)
            for j, pair in enumerate(self.dir_vec_pairs):
                joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
        else:
            assert False

        return joint_pos

    def get_joints(self, output_vecs):
        joints = self.convert_dir_vec_to_pose(output_vecs)
        return joints


def get_leg_loss(out, target, dist_coeff=0.3, vel_coeff=0.7, eps=1e-8):
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    loss_func = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=5.0)

    out_right_femur = out[RIGHT_FEMUR_BONE_IDX * 3:(RIGHT_FEMUR_BONE_IDX + 1) * 3]
    out_right_shin = out[RIGHT_SHIN_BONE_IDX * 3:(RIGHT_SHIN_BONE_IDX + 1) * 3]
    out_right_femur_mag = torch.linalg.norm(out_right_femur, dim=-1)
    out_right_shin_mag = torch.linalg.norm(out_right_shin, dim=-1)
    target_right_femur = target[RIGHT_FEMUR_BONE_IDX * 3:(RIGHT_FEMUR_BONE_IDX + 1) * 3]
    target_right_shin = target[RIGHT_SHIN_BONE_IDX * 3:(RIGHT_SHIN_BONE_IDX + 1) * 3]
    target_right_femur_mag = torch.linalg.norm(target_right_femur, dim=-1)
    target_right_shin_mag = torch.linalg.norm(target_right_shin, dim=-1)

    # cosine_dist_out_rleg = 1. - torch.einsum('btj, btj -> bt', out_right_femur, out_right_shin) / (out_right_femur_mag * out_right_shin_mag + eps)
    # cosine_dist_target_rleg =\
    #     1. - torch.einsum('btj, btj -> bt', target[..., RIGHT_FEMUR_BONE_IDX * 3:(RIGHT_FEMUR_BONE_IDX + 1) * 3], target[..., RIGHT_SHIN_BONE_IDX * 3:(RIGHT_SHIN_BONE_IDX + 1) * 3]) / (target_right_femur_mag * target_right_shin_mag + eps)
    cosine_dist_out_rleg = 1. - cosine_similarity(out_right_femur, out_right_shin)
    cosine_dist_target_rleg = 1. - cosine_similarity(target_right_femur, target_right_femur)
    # cosine_dist_out_rleg = torch.acos(torch.clamp(cosine_similarity(out_right_femur, out_right_shin), min=-1, max=1))
    # cosine_dist_target_rleg = torch.acos(torch.clamp(cosine_similarity(target_right_femur, target_right_femur), min=-1, max=1))
    cosine_dist_loss_rleg = dist_coeff * loss_func(cosine_dist_out_rleg, cosine_dist_target_rleg)
    cosine_vel_loss_rleg = vel_coeff * loss_func(cosine_dist_out_rleg[1:] - cosine_dist_out_rleg[:-1],
                                                 cosine_dist_target_rleg[1:] - cosine_dist_target_rleg[:-1])

    out_left_femur = out[LEFT_FEMUR_BONE_IDX * 3:(LEFT_FEMUR_BONE_IDX + 1) * 3]
    out_left_shin = out[LEFT_SHIN_BONE_IDX * 3:(LEFT_SHIN_BONE_IDX + 1) * 3]
    out_left_femur_mag = torch.linalg.norm(out_left_femur, dim=-1)
    out_left_shin_mag = torch.linalg.norm(out_left_shin, dim=-1)
    target_left_femur = target[LEFT_FEMUR_BONE_IDX * 3:(LEFT_FEMUR_BONE_IDX + 1) * 3]
    target_left_shin = target[LEFT_SHIN_BONE_IDX * 3:(LEFT_SHIN_BONE_IDX + 1) * 3]
    target_left_femur_mag = torch.linalg.norm(target_left_femur, dim=-1)
    target_left_shin_mag = torch.linalg.norm(target_left_shin, dim=-1)

    # cosine_dist_out_lleg = 1. - torch.einsum('btj, btj -> bt', out_left_femur, out_left_shin) / (out_left_femur_mag * out_left_shin_mag + eps)
    # cosine_dist_target_lleg =\
    #     1. - torch.einsum('btj, btj -> bt', target[..., LEFT_FEMUR_BONE_IDX * 3:(LEFT_FEMUR_BONE_IDX + 1) * 3], target[..., LEFT_SHIN_BONE_IDX * 3:(LEFT_SHIN_BONE_IDX + 1) * 3]) / (target_left_femur_mag * target_left_shin_mag + eps)
    cosine_dist_out_lleg = 1. - cosine_similarity(out_left_femur, out_left_shin)
    cosine_dist_target_lleg = 1. - cosine_similarity(target_left_femur, target_left_femur)
    # cosine_dist_out_lleg = torch.acos(torch.clamp(cosine_similarity(out_left_femur, out_left_shin), min=-1, max=1))
    # cosine_dist_target_lleg = torch.acos(torch.clamp(cosine_similarity(target_left_femur, target_left_femur), min=-1, max=1))
    cosine_dist_loss_lleg = dist_coeff * loss_func(cosine_dist_out_lleg, cosine_dist_target_lleg)
    cosine_vel_loss_lleg = vel_coeff * loss_func(cosine_dist_out_lleg[1:] - cosine_dist_out_lleg[:-1],
                                                 cosine_dist_target_lleg[1:] - cosine_dist_target_lleg[:-1])

    return cosine_dist_loss_rleg + cosine_vel_loss_rleg + cosine_dist_loss_lleg + cosine_vel_loss_lleg


# In[4]:
def get_ftct_loss(out, target, fk_routine):
    out_poses = fk_routine.get_joints(out)
    target_poses = fk_routine.get_joints(target)

    out_lf_speeds = torch.norm(out_poses[1:, LEFT_FOOT_JOINT_IDX] - out_poses[:-1, LEFT_FOOT_JOINT_IDX], dim=-1)[0]
    out_rf_speeds = torch.norm(out_poses[1:, RIGHT_FOOT_JOINT_IDX] - out_poses[:-1, RIGHT_FOOT_JOINT_IDX], dim=-1)[0]

    target_lf_speeds = torch.norm(target_poses[1:, LEFT_FOOT_JOINT_IDX] - target_poses[:-1, LEFT_FOOT_JOINT_IDX], dim=-1)
    target_rf_speeds = torch.norm(target_poses[1:, RIGHT_FOOT_JOINT_IDX] - target_poses[:-1, RIGHT_FOOT_JOINT_IDX], dim=-1)

    return torch.mean(torch.abs(target_lf_speeds - out_lf_speeds)) + torch.mean(torch.abs(target_rf_speeds - out_rf_speeds))


def get_velocity_loss(outputs, targets, loss_function, fk_routine): #will be defined as movement of poses from time1-time0, time2-time1, time3-time2 etc.

    batch_size, time_steps, pose_dim = outputs.shape

    total_loss = 0.

    for item in range(batch_size):
        out = outputs[item]
        target = targets[item]
        
        leg_loss = 0.
        leg_loss = get_leg_loss(out,target)
        ftct_loss = get_ftct_loss(out,target, fk_routine)

        non_zero_out = out
        non_zero_target = target

        out_1d = non_zero_out[2:] - non_zero_out[1:-1]
        target_1d = non_zero_target[2:] - non_zero_target[1:-1]

        v_loss = loss_function(out_1d, target_1d)
        p_loss = loss_function(non_zero_out, non_zero_target)

        total_loss += v_loss
        total_loss += p_loss
        total_loss += 0.3 * leg_loss
        total_loss += 0.002 * ftct_loss

    return total_loss

def forward_pass_s2ag(in_mfcc, in_chroma, label_in, pre_seq, target_poses, s2ag_generator, s2ag_discriminator, s2ag_gen_optimizer, s2ag_dis_optimizer, fk_routine):

    # make pre seq input
    # pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1],
    #                                     target_poses.shape[2] + 1))
    # pre_seq[:, 0:20, :-1] = target_poses[:, 0:20]
    # pre_seq[:, 0:20, -1] = 1  # indicating bit for constraints

    ###########################################################################################
    # train D
    dis_error = None

    s2ag_dis_optimizer.zero_grad()

    # out shape (batch x seq x dim)
    # s2s -> self, pre_seq, in_mfcc, in_chroma, label_val, target, teacher_force_ratio

    out_dir_vec = s2ag_generator(pre_seq, in_mfcc, in_chroma, label_in,target_poses, 0.5)
    print(out_dir_vec.shape)


    dis_real = s2ag_discriminator(target_poses, label_in)



    dis_fake = s2ag_discriminator(out_dir_vec.detach(), label_in)



    dis_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8)))  # ns-gan

    dis_error.backward()
    s2ag_dis_optimizer.step()

    ###########################################################################################
    # train G

    for _ in range(5):
        s2ag_gen_optimizer.zero_grad()

        # decoding

        out_dir_vec = s2ag_generator(pre_seq, in_mfcc, in_chroma, label_in,target_poses, 0.5)



        # loss
        beta = 0.1
        # huber_loss = F.smooth_l1_loss(out_dir_vec / beta, target_poses / beta) * beta
        
        #Define huber loss as the velocity loss thing
        loss_function = torch.nn.SmoothL1Loss()
        huber_loss = get_velocity_loss(out_dir_vec, target_poses, loss_function, fk_routine)


        dis_output = s2ag_discriminator(out_dir_vec, label_in)
        gen_error = -torch.mean(torch.log(dis_output + 1e-8))
        kld = div_reg = None

        loss_regression_weight = 500
        loss_gan_weight = 5.0

        loss = loss_regression_weight * huber_loss  # + var_loss

        loss += loss_gan_weight * gen_error

        loss.backward()
        s2ag_gen_optimizer.step()

    return F.l1_loss(out_dir_vec, target_poses).item(), dis_error.item(), loss.item() 

def per_train_epoch(s2ag_generator, s2ag_discriminator, s2ag_gen_optimizer, s2ag_dis_optimizer, fk_routine):

    s2ag_generator.train()
    s2ag_discriminator.train()
         
    batch_s2ag_loss = 0.

    dis_loss = 0.
    gen_loss = 0.

    # Use batch size of 8

    num_batches = len(train_dataloader)

    

    # Make a batch loop here

    for batch_idx, (pre_pose_vecs, target_pose_vecs, train_mfccs, train_chromas, train_labels, seq_name) in enumerate(train_dataloader):

        start_time = time.time()

        loss, dis_error, gen_error = forward_pass_s2ag(train_mfccs.to(device=device), \
                                 train_chromas.to(device=device), \
                                 train_labels.to(device=device), \
                                 pre_pose_vecs.to(device=device),\
                                 target_pose_vecs.to(device=device),\
                                 s2ag_generator, s2ag_discriminator, s2ag_gen_optimizer, s2ag_dis_optimizer, fk_routine)
        # Compute statistics
        batch_s2ag_loss += loss

        dis_loss+=dis_error
        gen_loss+=gen_error

        print("Batch: {}/{}, Time per batch: {}".format(batch_idx,num_batches,int(np.ceil(time.time() - start_time))))
    

    batch_s2ag_loss /= num_batches

    dis_loss /=num_batches
    gen_loss /= num_batches

    print("Mean batch loss: {}, Dis loss: {}, Gen loss: {}".format(batch_s2ag_loss, dis_loss, gen_loss))


from Model_Components.m2ad_pose_repletion_net import PoseSeq2SeqGenerator, AffDiscriminator

encoder_args = {
    'hidden_size':256,
    'n_layers':5,
    'dropout_prob':0.4
}

decoder_args = {
    'hidden_size':256,
    'n_layers':5,
    'dropout_prob':0.4
}
teacher_force_ratio = 0.5
device = 'cuda'
s2ag_generator = PoseSeq2SeqGenerator(encoder_args, decoder_args, 440, 37, 440, 12, 2, 100).to(device=device)
s2ag_discriminator = AffDiscriminator(2,100, 16).to(device=device)





def train(s2ag_generator,s2ag_discriminator):

    fk_routine = ReconstructSkeleton(device=device) 

    s2ag_start_epoch = 0

    lr_s2ag_gen = 1e-3
    lr_s2ag_dis = 1e-5

    # s2ag optimizers
    s2ag_gen_optimizer = optim.Adam(s2ag_generator.parameters(),
                                            lr=lr_s2ag_gen, 
                                    betas=(0.5, 0.999)
                                    )

    s2ag_dis_optimizer = torch.optim.Adam(
        s2ag_discriminator.parameters(),
        lr=lr_s2ag_dis,
        betas=(0.5, 0.999)
        )

    for epoch in range(s2ag_start_epoch, 500):

        if epoch%100 == 0 and epoch >= 300:

            lr_s2ag_gen /= 5
            lr_s2ag_dis /= 5

            # s2ag optimizers
            s2ag_gen_optimizer = optim.Adam(s2ag_generator.parameters(),
                                                    lr=lr_s2ag_gen, 
                                            betas=(0.5, 0.999)
                                            )

            s2ag_dis_optimizer = torch.optim.Adam(
                s2ag_discriminator.parameters(),
                lr=lr_s2ag_dis,
                betas=(0.5, 0.999)
                )

        # training
        print('s2ag training epoch: {}'.format(epoch))
        per_train_epoch(s2ag_generator,s2ag_discriminator,s2ag_gen_optimizer,s2ag_dis_optimizer,fk_routine)
        print('Done.')
        

        # save model and weights
        
        if epoch%10 == 0:
            torch.save(s2ag_generator.state_dict(), './BN_RN_v7/M2AD_Gen11_ep'+str(epoch)+'_10FPS')
            torch.save(s2ag_discriminator.state_dict(), './BN_RN_v7/M2AD_Disc11_ep'+str(epoch)+'_10FPS')


train(s2ag_generator,s2ag_discriminator)