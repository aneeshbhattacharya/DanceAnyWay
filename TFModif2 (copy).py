import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import math

import time
import numpy as np
from speech2affective_gestures.net.utils.graph import Graph
from speech2affective_gestures.net.utils.tgcn import STGraphConv, STGraphConvTranspose
import shutil

import librosa
from librosa.feature import mfcc
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

import random


# In[2]:

print("Script started...")

from Model_Components.m2ad_beat_generator_net import Seq2SeqGenerator

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


# In[2]:



from torch.utils.data import Dataset, DataLoader

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


# In[3]:


def check_for_nan(pre_seq, target_pose_in, mfcc_in, chroma_in):
    pre_seq_check = np.isfinite(pre_seq).all()
    target_pose_in_check = np.isfinite(target_pose_in).all()
    mfcc_in_check = np.isfinite(mfcc_in).all()
    chroma_in_check = np.isfinite(chroma_in).all()

    if pre_seq_check == False or target_pose_in_check == False or mfcc_in_check == False or chroma_in_check == False:
        print("INF in dataset")


# In[5]:


gan_data = CustomGANDataset(mfcc_files,chroma_files,sc_files,beats_files,target_pose_vecs,target_labels, s2s.to(device=device), seq_names)

training_processed_data = []

for idx, (pre_seq, target_pose_in, mfcc_in, chroma_in, label_in, seq_name) in enumerate(gan_data):
    check_for_nan(pre_seq, target_pose_in, mfcc_in, chroma_in)
    training_processed_data.append(gan_data[idx])

    train_dataloader = DataLoader(training_processed_data, batch_size=8, shuffle=True)    


# In[6]:



# ## Loss Functions

# In[5]:


# Direction Vectors
#right-knee = 5-6 should be index [15:18] 
#right-shin = 6-7 should be index [18:21]
#left-knee = 11-12 should be index [21:24]
#left-shin = 12-13 should be index [24:27]

def get_leg_loss(out,target):
    loss_func = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=5.0)
    error = 0.
    for _ in range(out.shape[0]):
        
        right_femur_out_actual = out[_][15:18] * 10
        right_shin_out_actual = out[_][18:21] * 10
        right_femur_target = target[_][15:18] * 10
        right_shin_target = target[_][18:21] * 10
        
        # with torch.no_grad():
        #     non_diff_right_femur_out = torch.round(right_femur_out_actual,decimals=2).to(device)
        #     non_diff_right_shin_out = torch.round(right_shin_out_actual,decimals=2).to(device)

        #     difference_right_femur = right_femur_out_actual - non_diff_right_femur_out
        #     difference_right_shin = right_shin_out_actual - non_diff_right_shin_out 

        # right_femur_out = right_femur_out_actual - difference_right_femur
        # right_shin_out = right_shin_out_actual - difference_right_shin

        right_femur_out = right_femur_out_actual
        right_shin_out = right_shin_out_actual

        mag_right_femur = torch.linalg.norm(right_femur_out)
        mag_right_shin = torch.linalg.norm(right_shin_out)
        
        angle_val = (torch.dot(right_femur_out,right_shin_out))/(mag_right_femur*mag_right_shin)
        out_right_angle = 2
        if angle_val<=1 and angle_val>=-1:
            out_right_angle = torch.acos(angle_val)
            
        mag_right_femur = torch.linalg.norm(right_femur_target)
        mag_right_shin = torch.linalg.norm(right_shin_target)

        target_right_angle = torch.acos((torch.dot(right_femur_target,right_shin_target))/(mag_right_femur*mag_right_shin))
            
        left_femur_out_actual = out[_][21:24] * 10
        left_shin_out_actual = out[_][24:27] * 10
        left_femur_target = target[_][21:24] * 10
        left_shin_target = target[_][24:27] * 10
        
        # with torch.no_grad():
        #     non_diff_left_femur_out = torch.round(left_femur_out_actual,decimals=2).to(device)
        #     non_diff_left_shin_out = torch.round(left_shin_out_actual,decimals=2).to(device)

        #     difference_left_femur = left_femur_out_actual - non_diff_left_femur_out
        #     difference_left_shin = left_shin_out_actual - non_diff_left_shin_out 

        # left_femur_out = left_femur_out_actual - difference_left_femur
        # left_shin_out = left_shin_out_actual - difference_left_shin

        left_femur_out = left_femur_out_actual
        left_shin_out = left_shin_out_actual

        mag_left_femur = torch.linalg.norm(left_femur_out)
        mag_left_shin = torch.linalg.norm(left_shin_out)
        
        angle_val = (torch.dot(left_femur_out,left_shin_out))/(mag_left_femur*mag_left_shin)
        out_left_angle = 2
        if angle_val<=1 and angle_val>=-1:
            out_left_angle = torch.acos(angle_val)

        mag_left_femur = torch.linalg.norm(left_femur_target)
        mag_left_shin = torch.linalg.norm(left_shin_target)
        
        target_left_angle = torch.acos((torch.dot(left_femur_target,left_shin_target))/(mag_left_femur*mag_left_shin))

        if type(out_left_angle) is int or type(out_right_angle) is int:
            print("Invalid loss here")
            print(out_left_angle)
            print(out_right_angle)
            print(angle_val)
            print(left_femur_out)
            print(left_shin_out)
            print(right_femur_out)
            print(right_shin_out)

            out_left_angle = prev_out_left_anlge
            out_right_angle = prev_out_right_angle
        #Smooth L1
        output_vals_mat = torch.hstack((out_right_angle,out_left_angle))/100

        # target_vals_mat = torch.from_numpy(np.array([target_right_angle,target_left_angle])/100).to(device='cuda')
        target_vals_mat = torch.hstack((target_right_angle,target_left_angle))/100
        error+= 0.3 * loss_func(output_vals_mat,target_vals_mat)
        
        #ALSO DO SOME KIND OF ANGULAR VELOCITY LOSS
        
        if _ == 0:
            prev_out_left_anlge = out_left_angle
            prev_out_right_angle = out_right_angle
            
            prev_target_left_angle = target_left_angle
            prev_target_right_angle = target_right_angle

        
        else:
            # This is second index. Track the angular velocity i.e current angle - prev angle and get a loss
            # and make this the more dominant loss
            
            delta_out_left = out_left_angle - prev_out_left_anlge
            delta_out_right = out_right_angle - prev_out_right_angle
            
            delta_target_left = target_left_angle - prev_target_left_angle
            delta_target_right = target_right_angle - prev_target_right_angle
            
            # output_vals_mat = torch.from_numpy(np.array([delta_out_left,delta_out_right])/100).to(device='cuda')
            # target_vals_mat = torch.from_numpy(np.array([delta_target_left,delta_target_right])/100).to(device='cuda')

            output_vals_mat = torch.hstack((delta_out_left,delta_out_right))/100
            target_vals_mat = torch.hstack((delta_target_left,delta_target_right))/100
            
            error+= 0.7 * loss_func(output_vals_mat,target_vals_mat)
        
            prev_out_left_anlge = out_left_angle
            prev_out_right_angle = out_right_angle
            
            prev_target_left_angle = target_left_angle
            prev_target_right_angle = target_right_angle
        
        
    return error/100


# In[6]:


def get_velocity_loss(outputs, targets, loss_function): #will be defined as movement of poses from time1-time0, time2-time1, time3-time2 etc.

    batch_size, time_steps, pose_dim = outputs.shape

    total_loss = 0.

    for item in range(batch_size):
        out = outputs[item]
        target = targets[item]
        
        leg_loss = 0.
        leg_loss = get_leg_loss(out,target)

        non_zero_out = out
        non_zero_target = target

        out_1d = non_zero_out[2:] - non_zero_out[1:-1]
        target_1d = non_zero_target[2:] - non_zero_target[1:-1]

        v_loss = loss_function(out_1d, target_1d)
        p_loss = loss_function(non_zero_out, non_zero_target)

        total_loss += v_loss
        total_loss += p_loss
        total_loss += 0.3*leg_loss

    return total_loss


# In[7]:


def forward_pass_s2ag(in_mfcc, in_chroma, label_in, pre_seq, target_poses, s2ag_generator, s2ag_discriminator, s2ag_gen_optimizer, s2ag_dis_optimizer, tf_ratio):

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

    out_dir_vec = s2ag_generator(pre_seq, in_mfcc, in_chroma, label_in,target_poses, tf_ratio)
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

        out_dir_vec = s2ag_generator(pre_seq, in_mfcc, in_chroma, label_in,target_poses, tf_ratio)



        # loss
        beta = 0.1
        # huber_loss = F.smooth_l1_loss(out_dir_vec / beta, target_poses / beta) * beta
        
        #Define huber loss as the velocity loss thing
        loss_function = torch.nn.SmoothL1Loss()
        huber_loss = get_velocity_loss(out_dir_vec, target_poses, loss_function)


        dis_output = s2ag_discriminator(out_dir_vec, label_in)
        gen_error = -torch.mean(torch.log(dis_output + 1e-8))
        kld = div_reg = None

        loss_regression_weight = 500
        loss_gan_weight = 5.0

        loss = loss_regression_weight * huber_loss  # + var_loss

        loss += loss_gan_weight * gen_error

        loss.backward()

        torch.nn.utils.clip_grad_norm_(s2ag_generator.parameters(), max_norm = 1)

        s2ag_gen_optimizer.step()

    return F.l1_loss(out_dir_vec, target_poses).item(), dis_error.item(), loss.item() 


# In[8]:


def per_train_epoch(s2ag_generator, s2ag_discriminator, s2ag_gen_optimizer, s2ag_dis_optimizer, tf_ratio):

    s2ag_generator.train()
    s2ag_discriminator.train()
         
    batch_s2ag_loss = 0.

    dis_loss = 0.
    gen_loss = 0.

    # Use batch size of 8

    num_batches = len(train_dataloader)

    start_time = time.time()

    # Make a batch loop here

    for batch_idx, (pre_pose_vecs, target_pose_vecs, train_mfccs, train_chromas, train_labels, seq_name) in enumerate(train_dataloader):


        loss, dis_error, gen_error = forward_pass_s2ag(train_mfccs.to(device=device),train_chromas.to(device=device),train_labels.to(device=device),pre_pose_vecs.to(device=device),                                 target_pose_vecs.to(device=device),                                 s2ag_generator, s2ag_discriminator, s2ag_gen_optimizer, s2ag_dis_optimizer, tf_ratio)
        # Compute statistics
        batch_s2ag_loss += loss

        dis_loss+=dis_error
        gen_loss+=gen_error

        print("Batch: {}/{}, Time per batch: {}".format(batch_idx,num_batches,int(np.ceil(time.time() - start_time))))
    

    batch_s2ag_loss /= num_batches

    dis_loss /=num_batches
    gen_loss /= num_batches

    print("Mean batch loss: {}, Dis loss: {}, Gen loss: {}".format(batch_s2ag_loss, dis_loss, gen_loss))


# In[9]:


# In[7]:



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
device = 'cuda'
s2ag_generator = PoseSeq2SeqGenerator(encoder_args, decoder_args, 440, 37, 440, 12, 2, 100).to(device=device)
s2ag_discriminator = AffDiscriminator(2,100, 16).to(device=device)


# In[8]:


# s2ag_generator.load_state_dict(torch.load('./BN_RN_TFModif/M2AD_Gen11_ep20_10FPS'))
# s2ag_discriminator.load_state_dict(torch.load('./BN_RN_TFModif/M2AD_Disc11_ep20_10FPS'))


# In[9]:


def train(s2ag_generator,s2ag_discriminator):

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
    
    tf_ratio = 1.0

    for epoch in range(s2ag_start_epoch, 500):

        if tf_ratio>0.5:
            if epoch%20 == 0 and epoch <100 and epoch >0:
                tf_ratio -=0.1

            elif epoch%40 ==0 and epoch >= 100 and tf_ratio>0:
                tf_ratio -=0.1

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
        per_train_epoch(s2ag_generator,s2ag_discriminator,s2ag_gen_optimizer,s2ag_dis_optimizer, tf_ratio)
        print('Done.')
        

        # save model and weights
        
        if epoch%10 == 0:
            torch.save(s2ag_generator.state_dict(), './BN_RN_TFModif/M2AD_Gen11_ep'+str(epoch)+'_10FPS')
            torch.save(s2ag_discriminator.state_dict(), './BN_RN_TFModif/M2AD_Disc11_ep'+str(epoch)+'_10FPS')


print("Training started...")
train(s2ag_generator,s2ag_discriminator)
