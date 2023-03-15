import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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

device = torch.device('cuda')

class MFCCEncoder(nn.Module):
    def __init__(self, mfcc_length, num_mfcc, time_steps):
        super().__init__()
        self.conv1 = nn.Conv1d(mfcc_length, 64, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 48, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(48)
        self.conv3 = nn.Conv1d(48, time_steps, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(time_steps)
        # self.conv4 = nn.Conv1d(48, time_steps, 3, padding=1)
        # self.batch_norm4 = nn.BatchNorm1d(time_steps)

        self.linear1 = nn.Linear(num_mfcc, 32)

        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, mfcc_data):
        x_01 = self.activation(self.batch_norm1(self.conv1(mfcc_data.permute(0, 2, 1))))
        x_02 = self.activation(self.batch_norm2(self.conv2(x_01)))
        x_03 = self.activation(self.batch_norm3(self.conv3(x_02)))
        # x_04 = self.activation(self.batch_norm4(self.conv4(x_03)))
        out = self.activation(self.linear1(x_03))
        return out

class ChromaEncoder(nn.Module):
    def __init__(self, chroma_length, num_chroma, time_steps):
        super().__init__()
        self.conv1 = nn.Conv1d(chroma_length, 64, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 48, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(48)
        self.conv3 = nn.Conv1d(48, time_steps, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(time_steps)
        # self.conv4 = nn.Conv1d(48, time_steps, 3, padding=1)
        # self.batch_norm4 = nn.BatchNorm1d(time_steps)

        self.linear1 = nn.Linear(num_chroma, 4)

        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, chroma_data):
        x_01 = self.activation(self.batch_norm1(self.conv1(chroma_data.permute(0, 2, 1))))
        x_02 = self.activation(self.batch_norm2(self.conv2(x_01)))
        x_03 = self.activation(self.batch_norm3(self.conv3(x_02)))
        # x_04 = self.activation(self.batch_norm4(self.conv4(x_03)))
        out = self.activation(self.linear1(x_03))
        return out

# MODIFIED TIKTOK DATASET CODE


class AffEncoder(nn.Module):
    def __init__(self, feat_out, coords=3):
        super().__init__()

        self.dir_vec_pairs = [
            (0,1,5), (0,2,5), (0,8,5), (0,5,5), (0,11,5),
            (5,6,5), (6,7,5), (11,12,5), (12,13,5),
            (2,3,5), (3,4,5), (8,9,5), (9,10,5),
            (1,14,5), (14,16,5), (1,15,5), (15,17,5)
        ]

        self.dir_edge_pairs = [
            (0,1), (0,2), (0,3), (0,4), (0,13), (0,15),
            (3,5), (5,6), (4,7), (7,8), (1,9), (9,10),
            (2,11), (11,12), (13,14), (15,16)
        ]

        self.body_parts_edge_idx = [

            [13,14,15,16], #head
            [0,3,4], #torso
            [1,9,10], #left arm
            [2,11,12], #right arm
            [5,6], #left leg
            [7,8] #right leg
        ]

        self.max_body_part_edges = 4
        self.body_parts_edge_pairs = [
            (1,0), (1,2), (1,3), 
            (1,4), (1,5)
        ]



        self.coords = coords
        self.feat_out = feat_out

        self.num_dir_vec_pairs = len(self.dir_vec_pairs)
        graph1 = Graph(self.num_dir_vec_pairs,
                       self.dir_edge_pairs,
                       strategy='spatial',
                       max_hop=2)
        self.A1 = torch.tensor(graph1.A,
                               dtype=torch.float32,
                            #    requires_grad=False).cuda()
                            requires_grad=False).to(device='cuda')

        self.num_body_parts = len(self.body_parts_edge_idx)
        graph2 = Graph(self.num_body_parts,
                       self.body_parts_edge_pairs,
                       strategy='spatial',
                       max_hop=2)
        self.A2 = torch.tensor(graph2.A,
                               dtype=torch.float32,
                            #    requires_grad=False).cuda()
                            requires_grad=False).to(device='cuda')

        spatial_kernel_size1 = 5
        temporal_kernel_size1 = 9
        kernel_size1 = (temporal_kernel_size1, spatial_kernel_size1)
        padding1 = ((kernel_size1[0] - 1) // 2, (kernel_size1[1] - 1) // 2)
        self.st_gcn1 = STGraphConv(coords, 16, self.A1.size(0), kernel_size1,
                                   stride=(1, 1), padding=padding1)
        self.batch_norm1 = nn.BatchNorm1d(16 * self.num_dir_vec_pairs)

        spatial_kernel_size2 = 3
        temporal_kernel_size2 = 9
        kernel_size2 = (temporal_kernel_size2, spatial_kernel_size2)
        padding2 = ((kernel_size2[0] - 1) // 2, (kernel_size2[1] - 1) // 2)


        #IMP CHANGES: Changed input chanels from 48 to 64 then used 80 at bn2 and conv3

        # self.st_gcn2 = STGraphConv(48, 16, self.A2.size(0), kernel_size2,
        #                            stride=(1, 1), padding=padding2)
        # self.batch_norm2 = nn.BatchNorm1d(16 * self.num_body_parts)

        self.st_gcn2 = STGraphConv(64, 16, self.A2.size(0), kernel_size2,
                                   stride=(1, 1), padding=padding2)

        self.batch_norm2 = nn.BatchNorm1d(16 * self.num_body_parts)
        # self.pre_conv = nn.Sequential(
        #     nn.Conv1d(input_size, 16, 3),
        #     nn.BatchNorm1d(16),
        #     nn.LeakyReLU(True),
        #     nn.Conv1d(16, 8, 3),
        #     nn.BatchNorm1d(8),
        #     nn.LeakyReLU(True),
        #     nn.Conv1d(8, 8, 3),
        # )
        kernel_size3 = 5
        padding3 = (kernel_size3 - 1) // 2
        self.conv3 = nn.Conv1d(16 * self.num_body_parts, 16, kernel_size3, padding=padding3)
        self.batch_norm3 = nn.BatchNorm1d(16)

        kernel_size4 = 3
        padding4 = (kernel_size4 - 1) // 2
        self.conv4 = nn.Conv1d(16, self.feat_out, kernel_size4, padding=padding4) #changed here to 16 from 8
        self.batch_norm4 = nn.BatchNorm1d(self.feat_out)

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, poses):
        n, t, jc = poses.shape
        j = jc // self.coords
        # poses = torch.cat((poses, torch.zeros_like(poses[..., 0:3])), dim=-1)

        poses = poses.reshape(n,t,-1,3)

        # feat1_out, _ = self.st_gcn1(poses.view(n, t, -1, 3).permute(0, 3, 1, 2), self.A1)

        # poses = torch.from_numpy(poses)
        poses = poses.to(dtype=torch.float32)

        # poses.type(torch.FloatTensor)

        feat1_out, _ = self.st_gcn1(poses.permute(0, 3, 1, 2), self.A1)

        f1 = feat1_out.shape[1]

        feat1_out_bn = self.batch_norm1(feat1_out.permute(0, 1, 3, 2).contiguous().
                                        view(n, -1, t)).view(n, -1, self.num_dir_vec_pairs, t).permute(0, 1, 3, 2)
        feat2_in = torch.zeros((n, t,
                                self.max_body_part_edges * f1,
                                # self.num_body_parts)).float().cuda()
                                self.num_body_parts)).float().to(device='cuda')
                                
        for idx, body_part_idx in enumerate(self.body_parts_edge_idx):
            feat2_in[..., :f1 * len(body_part_idx), idx] =\
                feat1_out_bn[..., body_part_idx].permute(0, 2, 1, 3).contiguous().view(n, t, -1)



        feat2_in = feat2_in.permute(0, 2, 1, 3)

        feat2_out, _ = self.st_gcn2(feat2_in, self.A2)

        feat2_out_bn = self.batch_norm2(feat2_out.permute(0, 1, 3, 2).contiguous().
                                        view(n, -1, t)).view(n, -1, self.num_body_parts, t).permute(0, 1, 3, 2)
        feat3_in = feat2_out_bn.permute(0, 2, 1, 3).contiguous().view(n, t, -1).permute(0, 2, 1)
        feat3_out = self.activation(self.batch_norm3(self.conv3(feat3_in)))
        feat4_out = self.activation(self.batch_norm4(self.conv4(feat3_out))).permute(0, 2, 1)

        return feat4_out


class PoseEncoder(nn.Module):
    def __init__(self, args, mfcc_length, num_mfcc, chroma_length, num_chroma, label_classes, time_steps):
        super().__init__()
        self.mfcc_feature_length = 32
        self.pose_feature_length = 16
        self.chroma_feature_length = 4
        # self.embed_vector_length = 2    

        self.in_size = self.mfcc_feature_length + self.pose_feature_length + self.chroma_feature_length 
        # + self.embed_vector_length

        self.audio_encoder = MFCCEncoder(mfcc_length, num_mfcc, time_steps)
        self.chroma_encoder = ChromaEncoder(chroma_length,num_chroma, time_steps)
        self.aff_encoder = AffEncoder(self.pose_feature_length)

        self.hidden_size = args['hidden_size']
        self.lstm = nn.LSTM(self.in_size, hidden_size=self.hidden_size, num_layers=args['n_layers'], batch_first=True,
                          bidirectional=True, dropout=args['dropout_prob'])

        self.fc_hidden = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc_cell = nn.Linear(self.hidden_size*2, self.hidden_size)

    def forward(self, pre_seq, in_mfcc, in_chroma, label_val):

        audio_feat_seq = self.audio_encoder(in_mfcc)  # output (bs, n_frames, feat_size)

        chroma_feat_seq = self.chroma_encoder(in_chroma)

        pre_feat_seq = self.aff_encoder(pre_seq[..., :-1])

        in_data = torch.cat((pre_feat_seq, audio_feat_seq, chroma_feat_seq), dim=2)

        encoder_states, (hidden,cell) = self.lstm(in_data) # Hidden state of this lstm is now 6,8,256 (layers, batch, seq) -> 3,8,256

        # print("Encoder hidden",hidden.shape)


        hidden = self.fc_hidden(torch.cat((hidden[0:3], hidden[3:6]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:3], cell[3:6]), dim=2))

        # print("Encoder hidden",hidden.shape)

        return encoder_states, hidden, cell

class PoseDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pose_dim = 51
        self.in_size = 16

        self.pose_feature_length = 16

        self.hidden_size = args['hidden_size'] # set to 64 set n_layers as 2

        self.aff_encoder = AffEncoder(self.pose_feature_length)

        self.lstm = nn.LSTM(self.hidden_size*2 + self.in_size, hidden_size=self.hidden_size, num_layers=args['n_layers'], batch_first=True,
                          bidirectional=False, dropout=args['dropout_prob'])
        
        self.hidden_flattener = nn.Linear(self.hidden_size*3, self.hidden_size) # Use to convert 3,8,256 to 1,8,256 for hidden state
        
        self.energy = nn.Linear(self.hidden_size*3,1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size//2, self.pose_dim)
        )

    def forward(self, in_data, encoder_states, hidden, cell):

        # print("In Data Shape",in_data.shape)

        aff_encoded_data = self.aff_encoder(in_data)

        # print("Hidden shape", hidden.shape) #3, 8, 256 -> change to 8,3,256
        # print("Cell shape", cell.shape) #3, 8, 256 -> change to 8,3,256

        hidden_permuted = torch.cat((hidden[0:1],hidden[1:2]),dim=2)

        hidden_permuted = torch.cat((hidden_permuted,hidden[2:3]),dim=2)
        # print("Hidden permuted", hidden_permuted.shape)

        flattened_hidden = self.hidden_flattener(hidden_permuted).permute(1,0,2)

        # print("Flattened hidden", flattened_hidden.shape)

        # hidden = hidden.permute(1,0,2)

        # print("Hidden shape", hidden.shape)

        sequence_length = encoder_states.shape[1]
        h_reshaped = flattened_hidden.repeat(1,sequence_length,1)

        # print("h_reshaped", h_reshaped.shape)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states),dim=2)))

        attention = self.softmax(energy)
        # N, seq_len, 1 -> N, 1, seq_len

        attention = attention.permute(0,2,1) # N ,1 seq_length
        encoder_states = encoder_states # N, seq_length, hidden_size

        context_vector = torch.bmm(attention, encoder_states)

        # print("Context Vector shape", context_vector.shape)
        

        rnn_input = torch.cat((context_vector, aff_encoded_data),dim=2)
        
        # hidden = hidden.permute(1,0,2)

        # print("RNN Input shape",rnn_input.shape)

        outputs, (hidden,cell) = self.lstm(rnn_input,(hidden,cell))

        out = self.out(outputs)
        return out, hidden, cell

class Seq2SeqGenerator(nn.Module):
    def __init__(self, args_encoder, args_decoder, mfcc_length, num_mfcc, chroma_length, num_chroma, label_classes, time_steps):
        super().__init__()
        self.encoder = PoseEncoder(args_encoder, mfcc_length, num_mfcc, chroma_length, num_chroma, label_classes, time_steps)
        self.decoder = PoseDecoder(args_decoder)

    def forward(self, pre_seq, in_mfcc, in_chroma, label_val, target, teacher_force_ratio):

        outputs = torch.zeros(target.shape[0],target.shape[1],target.shape[2]).to(device='cuda')

        encoder_states, hidden, cell = self.encoder(pre_seq, in_mfcc, in_chroma, label_val)


        input = target[:,0,:] #first time step of each batch

        input = input.unsqueeze(1)

        outputs[:,0,:] = input.squeeze(1)

        for t in range(1, target.shape[1]):
            output, hidden, cell = self.decoder(input, encoder_states, hidden, cell)

            outputs[:,t,:] = output.squeeze(1) 

            input = target[:,t,:].unsqueeze(1) if random.random() < teacher_force_ratio else output

        return outputs
