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


class MFCCEncoder(nn.Module):
    def __init__(self, mfcc_length, num_mfcc, time_steps):
        super().__init__()
        self.conv1 = nn.Conv1d(mfcc_length, 128, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 90, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(90)
        self.conv4 = nn.Conv1d(90, time_steps, 3, padding=1)
        self.batch_norm4 = nn.BatchNorm1d(time_steps)

        self.linear1 = nn.Linear(num_mfcc, 32)

        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, mfcc_data):
        x_01 = self.activation(self.batch_norm1(self.conv1(mfcc_data.permute(0, 2, 1))))
        x_02 = self.activation(self.batch_norm2(self.conv2(x_01)))
        x_03 = self.activation(self.batch_norm3(self.conv3(x_02)))
        x_04 = self.activation(self.batch_norm4(self.conv4(x_03)))
        out = self.activation(self.linear1(x_04))
        return out

class ChromaEncoder(nn.Module):
    def __init__(self, chroma_length, num_chroma, time_steps):
        super().__init__()
        self.conv1 = nn.Conv1d(chroma_length, 128, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 90, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(90)
        self.conv4 = nn.Conv1d(90, time_steps, 3, padding=1)
        self.batch_norm4 = nn.BatchNorm1d(time_steps)

        self.linear1 = nn.Linear(num_chroma, 6)

        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, chroma_data):
        x_01 = self.activation(self.batch_norm1(self.conv1(chroma_data.permute(0, 2, 1))))
        x_02 = self.activation(self.batch_norm2(self.conv2(x_01)))
        x_03 = self.activation(self.batch_norm3(self.conv3(x_02)))
        x_04 = self.activation(self.batch_norm4(self.conv4(x_03)))
        out = self.activation(self.linear1(x_04))
        return out

class AffEncoder(nn.Module):
    def __init__(self, pose_feat_length, coords=3):
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
        self.pose_feat_length = pose_feat_length
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
        self.conv4 = nn.Conv1d(16, self.pose_feat_length, kernel_size4, padding=padding4)
        self.batch_norm4 = nn.BatchNorm1d(self.pose_feat_length)

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
        self.chroma_feature_length = 6
#         self.embed_vector_length = 6    

        self.in_size = self.mfcc_feature_length + self.pose_feature_length + self.chroma_feature_length 
        #+ self.embed_vector_length

        self.audio_encoder = MFCCEncoder(mfcc_length, num_mfcc, time_steps)
        self.chroma_encoder = ChromaEncoder(chroma_length,num_chroma, time_steps)
        self.aff_encoder = AffEncoder(self.pose_feature_length)

        self.hidden_size = args['hidden_size']
        self.lstm = nn.LSTM(self.in_size, hidden_size=self.hidden_size, num_layers=args['n_layers'], batch_first=True,
                          bidirectional=False, dropout=args['dropout_prob'])

#         self.embed = nn.Embedding(label_classes,self.embed_vector_length)

    def forward(self, pre_seq, in_mfcc, in_chroma, label_val):

        # decoder_hidden = None
        # if self.do_flatten_parameters:
        #     self.gru.flatten_parameters()

        audio_feat_seq = self.audio_encoder(in_mfcc)  # output (bs, n_frames, feat_size)

        chroma_feat_seq = self.chroma_encoder(in_chroma)

        pre_feat_seq = self.aff_encoder(pre_seq[..., :-1])

        in_data = torch.cat((pre_feat_seq, audio_feat_seq, chroma_feat_seq), dim=2)

#         embed_out = self.embed(label_val.permute(1,0)).permute(1,0,2) 
#         repeated_embed = embed_out.repeat(1,in_data.shape[1],1)
        

#         in_data = torch.cat((in_data,repeated_embed),dim=2)

        output, (hidden,cell) = self.lstm(in_data)

        # print(hidden.shape)

        # hidden = hidden[:, :, :self.hidden_size] + hidden[:, :, self.hidden_size:]

        return hidden,cell

class PoseDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pose_dim = 51
        self.in_size = 16

        self.pose_feature_length = 16
        self.hidden_size = args['hidden_size'] # set to 64 set n_layers as 2

        self.aff_encoder = AffEncoder(self.pose_feature_length)

        self.lstm = nn.LSTM(self.in_size, hidden_size=self.hidden_size, num_layers=args['n_layers'], batch_first=True,
                          bidirectional=False, dropout=args['dropout_prob'])

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LayerNorm(self.hidden_size//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size//2, self.pose_dim)
        )

    def forward(self, in_data, hidden, cell):

        aff_encoded_data = self.aff_encoder(in_data)

        # mfcc_data = torch.rand(8,1,32)
        # chroma_data = torch.rand(8,1,6)
        # embed_data = torch.rand(8,1,6)

        # print(aff_encoded_data.shape)

        # aff_encoded_data = torch.cat((aff_encoded_data, mfcc_data, chroma_data, embed_data),dim=2)

        outputs, (hidden,cell) = self.lstm(aff_encoded_data,(hidden,cell))

        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        out = self.out(outputs)
        return out, hidden, cell

class PoseSeq2SeqGenerator(nn.Module):
    def __init__(self, args_encoder, args_decoder, mfcc_length, num_mfcc, chroma_length, num_chroma, label_classes, time_steps):
        super().__init__()
        self.encoder = PoseEncoder(args_encoder, mfcc_length, num_mfcc, chroma_length, num_chroma, label_classes, time_steps)
        self.decoder = PoseDecoder(args_decoder)


    def forward(self, pre_seq, in_mfcc, in_chroma, label_val, target, teacher_force_ratio):

        outputs = torch.zeros(target.shape[0],target.shape[1],target.shape[2]).to(device='cuda')
        hidden_list = []
        cell_list = []
        # print(outputs.shape)

        hidden, cell = self.encoder(pre_seq, in_mfcc, in_chroma, label_val)

        hidden_original = hidden

        input = target[:,0,:] #first time step of each batch

        input = input.unsqueeze(1)

        outputs[:,0,:] = input.squeeze(1)

        hidden_list.append(hidden)
        cell_list.append(cell)
        
        for t in range(1, target.shape[1]):
            

            output, hidden, cell = self.decoder(input, hidden*0.6 + hidden_original *0.4, cell)
            hidden_list.append(hidden)
            cell_list.append(cell)
            outputs[:,t,:] = output.squeeze(1) 

            if t<20:
                input = target[:,t,:].unsqueeze(1)  #utilizes the 1st 3 input beats with more probability

            else:
                input = target[:,t,:].unsqueeze(1) if random.random() < teacher_force_ratio else output

#             input = target[:,t,:].unsqueeze(1) if random.random() < teacher_force_ratio else output

        return outputs, hidden_list, cell_list
        
class AffDiscriminator(nn.Module):
    def __init__(self, label_classes, time_steps, pose_feat_length, coords=3):
        super().__init__()

        self.coords = coords
        self.hidden_size = 64

#         self.embed_vector_length = 6

        self.pose_feat_length = pose_feat_length

        self.aff_encoder = AffEncoder(self.pose_feat_length, coords=coords)

        self.gru = nn.GRU(self.pose_feat_length, hidden_size=self.hidden_size,
                          num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        

        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(time_steps, 1)

        self.activation = nn.LeakyReLU(inplace=True)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

#         self.embed = nn.Embedding(label_classes,self.embed_vector_length)

    def forward(self, poses, label_val, in_text=None):
        n, t, jc = poses.shape
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        aff_encoder_out = self.aff_encoder(poses)
        output_gru, decoder_hidden = self.gru(aff_encoder_out, decoder_hidden)

        # sum bidirectional outputs
        output_gru_bi = output_gru[:, :, :self.hidden_size] + output_gru[:, :, self.hidden_size:]



        # apply linear to every output
        output_linear1 = self.out(output_gru_bi.contiguous().view(-1, output_gru_bi.shape[2])).view(n, -1)

        # permuted_embed = label_val.contiguous().view(n,-1)

#         embed_out = self.embed(label_val.permute(1,0)).permute(1,0,2) 
#         permuted_embed = embed_out.contiguous().view(n,-1)


#         concated_output = torch.cat((output_linear1,permuted_embed),dim=1)

        output_linear2 = self.out2(output_linear1)
#         output_linear2 = self.out2(concated_output)

        return torch.sigmoid(output_linear2)