from typing import Optional

import torch.nn as nn
from torch import Tensor


def get_leg_loss(
    out: Tensor, 
    target: Tensor, 
    dist_coeff: Optional[float]=0.3, 
    vel_coeff: Optional[float]=0.7, 
) -> Tensor:
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    loss_func = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=5.0)
    out_right_femur = out[:, 5 * 3:(5 + 1) * 3]
    out_right_shin = out[:, 6 * 3:(6 + 1) * 3]
    target_right_femur = target[:, 5 * 3:(5 + 1) * 3]
    cosine_dist_out_rleg = 1. - cosine_similarity(out_right_femur, out_right_shin)
    cosine_dist_target_rleg = 1. - cosine_similarity(target_right_femur, target_right_femur)
    cosine_dist_loss_rleg = dist_coeff * loss_func(cosine_dist_out_rleg, cosine_dist_target_rleg)
    cosine_vel_loss_rleg = vel_coeff * loss_func(
        cosine_dist_out_rleg[1:] - cosine_dist_out_rleg[:-1],
        cosine_dist_target_rleg[1:] - cosine_dist_target_rleg[:-1])

    out_left_femur = out[:, 7 * 3:(7 + 1) * 3]
    out_left_shin = out[:, 8 * 3:(8 + 1) * 3]
    target_left_femur = target[:, 7 * 3:(7 + 1) * 3]
    cosine_dist_out_lleg = 1. - cosine_similarity(out_left_femur, out_left_shin)
    cosine_dist_target_lleg = 1. - cosine_similarity(target_left_femur, target_left_femur)
    cosine_dist_loss_lleg = dist_coeff * loss_func(cosine_dist_out_lleg, cosine_dist_target_lleg)
    cosine_vel_loss_lleg = vel_coeff * loss_func(
        cosine_dist_out_lleg[1:] - cosine_dist_out_lleg[:-1],
        cosine_dist_target_lleg[1:] - cosine_dist_target_lleg[:-1])
    
    return cosine_dist_loss_rleg + cosine_vel_loss_rleg + cosine_dist_loss_lleg + cosine_vel_loss_lleg

def get_velocity_loss(
    outputs: Tensor, 
    targets: Tensor, 
loss_function) -> Tensor:
    batch_size, _, _ = outputs.shape
    total_loss = 0.
    for item in range(batch_size):
        out = outputs[item]
        target = targets[item]
        
        leg_loss = 0.
        leg_loss = get_leg_loss(out,target)
        non_zero_out = out
        non_zero_target = target
        out_1d = non_zero_out[1:] - non_zero_out[:-1]
        target_1d = non_zero_target[1:] - non_zero_target[:-1]
        v_loss = loss_function(out_1d, target_1d) 
        p_loss = loss_function(non_zero_out, non_zero_target)

        total_loss += v_loss
        total_loss += p_loss
        total_loss += 0.3 * leg_loss

    return total_loss
