import torch
import torch.nn as nn
from torch import Tensor

from utils.custom_encoders import (
    MFCCEncoder,
    ChromaEncoder,
    AffEncoder,
)


class BeatPoseGenerator(nn.Module):
    def __init__(
        self,
        device: str,
        mfcc_length: int,
        num_mfcc: int,
        chroma_length: int,
        num_chroma: int,
        time_steps: int,
        noise_dim: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.mfcc_feats_length = 32
        self.chroma_feats_length = 6
        self.pose_feats_length = 16
        self.noise_dim = noise_dim

        self.in_size = (
            self.mfcc_feats_length
            + self.chroma_feats_length
            + self.pose_feats_length
            + self.noise_dim
        )

        self.mfcc_encoder = MFCCEncoder(mfcc_length, num_mfcc, time_steps)
        self.chroma_encoder = ChromaEncoder(chroma_length, num_chroma, time_steps)
        self.aff_encoder = AffEncoder(self.pose_feats_length, self.device)

        self.linear1 = nn.Linear(self.in_size, 84)
        self.encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=84, nhead=4, batch_first=True, dim_feedforward=512
        )
        self.linear2 = nn.Linear(84, 72)
        self.encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=72, nhead=4, batch_first=True, dim_feedforward=512
        )
        self.linear3 = nn.Linear(72, 64)
        self.encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, batch_first=True, dim_feedforward=512
        )
        self.linear4 = nn.Linear(64, 56)
        self.encoder_layer4 = nn.TransformerEncoderLayer(
            d_model=56, nhead=4, batch_first=True, dim_feedforward=512
        )
        self.linear5 = nn.Linear(56, 51)
        self.encoder_layer5 = nn.TransformerEncoderLayer(
            d_model=51, nhead=3, batch_first=True, dim_feedforward=512
        )
        self.linear6 = nn.Linear(51, 51)

    def forward(
        self,
        pre_seq: Tensor,
        mfcc: Tensor,
        chroma: Tensor,
    ) -> Tensor:
        batch_size = pre_seq.shape[0]
        time_steps = pre_seq.shape[1]

        noise = torch.randn(batch_size, time_steps, self.noise_dim).to(self.device)
        mfcc_feats = self.mfcc_encoder(mfcc)
        chroma_feats = self.chroma_encoder(chroma)
        pre_seq_feats = self.aff_encoder(pre_seq)
        
        encoder_features = torch.cat(
            (noise, pre_seq_feats, mfcc_feats, chroma_feats), dim=2
        )

        beat_poses = self.linear1(encoder_features)
        beat_poses = self.encoder_layer1(beat_poses)
        beat_poses = self.linear2(beat_poses)
        beat_poses = self.encoder_layer2(beat_poses)
        beat_poses = self.linear3(beat_poses)
        beat_poses = self.encoder_layer3(beat_poses)
        beat_poses = self.linear4(beat_poses)
        beat_poses = self.encoder_layer4(beat_poses)
        beat_poses = self.linear5(beat_poses)
        beat_poses = self.encoder_layer5(beat_poses)
        beat_poses = self.linear6(beat_poses)

        return beat_poses


def get_beat_pose_generator(config: dict) -> BeatPoseGenerator:
    try:
        generator = BeatPoseGenerator(
            config["device"],
            config["mfcc_length"],
            config["num_mfcc"],
            config["chroma_length"],
            config["num_chroma"],
            config["time_steps"],
            config["noise_dim"],
        )
        generator = generator.to(config["device"])
    except Exception as e:
        raise ValueError(f"Error creating beat pose generator: {e}")

    return generator
