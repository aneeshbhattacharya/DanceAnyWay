import argparse
import os
import yaml

import torch

from dataset import get_beat_pose_dataloader
from beat_pose_generator import get_beat_pose_generator
from utils.loss_functions import get_velocity_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="beat_pose_config.yaml")

    return parser.parse_args()


def train(config: dict) -> None:
    generator = get_beat_pose_generator(config)
    generator.train()

    data_path = config["dataset"]["train"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    save_dir = config["save_dir"]
    loss_save_path = config["loss_save_path"]
    save_path = os.path.join(save_dir, "checkpoint.pt")
    frame_loss_weight = config["frame_loss_weight"]
    device = config["device"]
    
    dataloader = get_beat_pose_dataloader(data_path, batch_size)
    
    print(f'Dataloader loaded with {len(dataloader)} batches')
    
    betas = (0.5, 0.99)

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    loss_function = torch.nn.SmoothL1Loss()

    if loss_save_path is not None:
        with open(loss_save_path, "w") as f:
            f.write(f"Epoch\tBatch\tLoss\n")
            
    

    for epoch in range(num_epochs):
        for idx, (pre_seqs, targets, mfccs, chromas, _, _) in enumerate(
            dataloader
        ):
            pre_seqs = pre_seqs.to(device)
            mfccs = mfccs.to(device)
            chromas = chromas.to(device)
            targets = targets.to(device)

            predicted_poses = generator(pre_seqs, mfccs, chromas)

            loss = (
                get_velocity_loss(predicted_poses, targets, loss_function)
                * frame_loss_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_save_path is not None:
                with open(loss_save_path, "a") as f:
                    f.write(f"{epoch}\t{idx}\t{loss.item()}\n")

        if loss_save_path is not None:
            with open(loss_save_path, "a") as f:
                f.write(
                    f"_____________________________________________________________\n"
                )

        if epoch % 10 == 0:
            curr_gen_save_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
            torch.save(generator.state_dict(), curr_gen_save_path)

        torch.save(generator.state_dict(), save_path)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())

    train(config)
