from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .graph import Graph
from .tgcn import STGraphConv


class MFCCEncoder(nn.Module):
    def __init__(self, mfcc_length: int, num_mfcc: int, time_steps: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(mfcc_length, 128, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 90, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(90)
        self.conv4 = nn.Conv1d(90, time_steps, 3, padding=1)
        self.batch_norm4 = nn.BatchNorm1d(time_steps)

        self.linear = nn.Linear(num_mfcc, 32)
        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, mfcc: Tensor) -> Tensor:
        mfcc = mfcc.permute(0, 2, 1)
        mfcc_feats = self.activation(self.batch_norm1(self.conv1(mfcc)))
        mfcc_feats = self.activation(self.batch_norm2(self.conv2(mfcc_feats)))
        mfcc_feats = self.activation(self.batch_norm3(self.conv3(mfcc_feats)))
        mfcc_feats = self.activation(self.batch_norm4(self.conv4(mfcc_feats)))
        mfcc_feats = self.activation(self.linear(mfcc_feats))
        return mfcc_feats


class ChromaEncoder(nn.Module):
    def __init__(self, chroma_length: int, num_chroma: int, time_steps: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(chroma_length, 128, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 90, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(90)
        self.conv4 = nn.Conv1d(90, time_steps, 3, padding=1)
        self.batch_norm4 = nn.BatchNorm1d(time_steps)

        self.linear = nn.Linear(num_chroma, 6)
        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, chroma: Tensor) -> Tensor:
        chroma = chroma.permute(0, 2, 1)
        chroma_feats = self.activation(self.batch_norm1(self.conv1(chroma)))
        chroma_feats = self.activation(self.batch_norm2(self.conv2(chroma_feats)))
        chroma_feats = self.activation(self.batch_norm3(self.conv3(chroma_feats)))
        chroma_feats = self.activation(self.batch_norm4(self.conv4(chroma_feats)))
        chroma_feats = self.activation(self.linear(chroma_feats))
        return chroma_feats


class AffEncoder(nn.Module):
    def __init__(
        self, noise_dim: int, device: str, coords: Optional[int] = 3
    ) -> None:
        super().__init__()

        self.device = device

        self.dir_vec_pairs = [
            (0, 1, 5),
            (0, 2, 5),
            (0, 8, 5),
            (0, 5, 5),
            (0, 11, 5),
            (5, 6, 5),
            (6, 7, 5),
            (11, 12, 5),
            (12, 13, 5),
            (2, 3, 5),
            (3, 4, 5),
            (8, 9, 5),
            (9, 10, 5),
            (1, 14, 5),
            (14, 16, 5),
            (1, 15, 5),
            (15, 17, 5),
        ]
        self.dir_edge_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 13),
            (0, 15),
            (3, 5),
            (5, 6),
            (4, 7),
            (7, 8),
            (1, 9),
            (9, 10),
            (2, 11),
            (11, 12),
            (13, 14),
            (15, 16),
        ]
        self.body_parts_edge_idx = [
            [13, 14, 15, 16],  # head
            [0, 3, 4],  # torso
            [1, 9, 10],  # left arm
            [2, 11, 12],  # right arm
            [5, 6],  # left leg
            [7, 8],  # right leg
        ]
        self.body_parts_edge_pairs = [(1, 0), (1, 2), (1, 3), (1, 4), (1, 5)]
        self.max_body_part_edges = 4
        self.coords = coords
        self.noise_dim = noise_dim
        self.num_dir_vec_pairs = len(self.dir_vec_pairs)

        graph1 = Graph(
            self.num_dir_vec_pairs, self.dir_edge_pairs, strategy="spatial", max_hop=2
        )
        self.A1 = torch.tensor(graph1.A, dtype=torch.float32, requires_grad=False).to(
            device=self.device
        )

        self.num_body_parts = len(self.body_parts_edge_idx)
        graph2 = Graph(
            self.num_body_parts,
            self.body_parts_edge_pairs,
            strategy="spatial",
            max_hop=2,
        )
        self.A2 = torch.tensor(graph2.A, dtype=torch.float32, requires_grad=False).to(
            device=self.device
        )

        spatial_kernel_size1 = 5
        temporal_kernel_size1 = 9
        kernel_size1 = (temporal_kernel_size1, spatial_kernel_size1)
        padding1 = ((kernel_size1[0] - 1) // 2, (kernel_size1[1] - 1) // 2)
        self.st_gcn1 = STGraphConv(
            coords, 16, self.A1.size(0), kernel_size1, stride=(1, 1), padding=padding1
        )
        self.batch_norm1 = nn.BatchNorm1d(16 * self.num_dir_vec_pairs)

        spatial_kernel_size2 = 3
        temporal_kernel_size2 = 9
        kernel_size2 = (temporal_kernel_size2, spatial_kernel_size2)
        padding2 = ((kernel_size2[0] - 1) // 2, (kernel_size2[1] - 1) // 2)

        self.st_gcn2 = STGraphConv(
            64, 16, self.A2.size(0), kernel_size2, stride=(1, 1), padding=padding2
        )

        self.batch_norm2 = nn.BatchNorm1d(16 * self.num_body_parts)
        kernel_size3 = 5
        padding3 = (kernel_size3 - 1) // 2
        self.conv3 = nn.Conv1d(
            16 * self.num_body_parts, 16, kernel_size3, padding=padding3
        )
        self.batch_norm3 = nn.BatchNorm1d(16)

        kernel_size4 = 3
        padding4 = (kernel_size4 - 1) // 2
        self.conv4 = nn.Conv1d(16, self.noise_dim, kernel_size4, padding=padding4)
        self.batch_norm4 = nn.BatchNorm1d(self.noise_dim)

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, poses: Tensor) -> Tensor:
        n, t, _ = poses.shape
        poses = poses.reshape(n, t, -1, 3)
        poses = poses.to(dtype=torch.float32)

        feat1_out, _ = self.st_gcn1(poses.permute(0, 3, 1, 2), self.A1)

        f1 = feat1_out.shape[1]

        feat1_out_bn = (
            self.batch_norm1(feat1_out.permute(0, 1, 3, 2).contiguous().view(n, -1, t))
            .view(n, -1, self.num_dir_vec_pairs, t)
            .permute(0, 1, 3, 2)
        )
        feat2_in = (
            torch.zeros((n, t, self.max_body_part_edges * f1, self.num_body_parts))
            .float()
            .to(device="cuda")
        )

        for idx, body_part_idx in enumerate(self.body_parts_edge_idx):
            feat2_in[..., : f1 * len(body_part_idx), idx] = (
                feat1_out_bn[..., body_part_idx]
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(n, t, -1)
            )

        feat2_in = feat2_in.permute(0, 2, 1, 3)

        feat2_out, _ = self.st_gcn2(feat2_in, self.A2)

        feat2_out_bn = (
            self.batch_norm2(feat2_out.permute(0, 1, 3, 2).contiguous().view(n, -1, t))
            .view(n, -1, self.num_body_parts, t)
            .permute(0, 1, 3, 2)
        )
        feat3_in = (
            feat2_out_bn.permute(0, 2, 1, 3)
            .contiguous()
            .view(n, t, -1)
            .permute(0, 2, 1)
        )
        feat3_out = self.activation(self.batch_norm3(self.conv3(feat3_in)))
        feat4_out = self.activation(self.batch_norm4(self.conv4(feat3_out))).permute(
            0, 2, 1
        )

        return feat4_out
