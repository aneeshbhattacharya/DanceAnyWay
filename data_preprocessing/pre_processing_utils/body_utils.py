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


aist_convention = {
    'nose':0,
    'l_eye':1,
    'r_eye':2,
    'l_ear':3,
    'r_ear':4,
    'l_shoulder':5,
    'r_shoulder':6,
    'l_elbow':7,
    'r_elbow':8,
    'l_wrist':9,
    'r_wrist':10,
    'l_hip':11,
    'r_hip':12,
    'l_knee':13,
    'r_knee':14,
    'l_ankle':15,
    'r_ankle':16
}

m2ad_convention ={
    'neck':0,
    'nose':1,
    'l_eye':15,
    'r_eye':14,
    'l_ear':17,
    'r_ear':16,
    'l_shoulder':2,
    'r_shoulder':8,
    'l_elbow':3,
    'r_elbow':9,
    'l_wrist':4,
    'r_wrist':10,
    'l_hip':5,
    'r_hip':11,
    'l_knee':6,
    'r_knee':12,
    'l_ankle':7,
    'r_ankle':13
}

def convert_aist_to_m2ad(aist_poses):
    #aist_pose_is_of_shape 17,3 assuming x,y,z
    #m2ad needs 18,3 in z,x,y
    
    m2ad_sequence = []
    
    for _ in range(aist_poses.shape[0]):
        
        aist_pose = aist_poses[_]
    
        m2ad_pose = np.zeros((18,3))

        for aist_body_part in aist_convention.keys():

            aist_index = aist_convention[aist_body_part]

            aist_values = aist_pose[aist_index]

            m2ad_values = np.array([aist_values[2],aist_values[0],aist_values[1]]) 
            
            m2ad_bodypart_index = m2ad_convention[aist_body_part]

            m2ad_pose[m2ad_bodypart_index] = m2ad_values

        # Get the neck joint position as the mid of aist left and right shoulder

        aist_left_shoulder = aist_pose[5]
        aist_right_shoulder = aist_pose[6]

        mid = (aist_left_shoulder + aist_right_shoulder)/2

        m2ad_pose[0] = m2ad_values = np.array([mid[2],mid[0],mid[1]]) 

        m2ad_sequence.append(m2ad_pose)
        
    return np.array(m2ad_sequence)
    

dir_vec_pairs = [
    (0,1,5), (0,2,5), (0,8,5), (0,5,5), (0,11,5),
    (5,6,5), (6,7,5), (11,12,5), (12,13,5),
    (2,3,5), (3,4,5), (8,9,5), (9,10,5),
    (1,14,5), (14,16,5), (1,15,5), (15,17,5)
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

def get_vec(pose,dir_vec_pairs):
    dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
    for i, pair in enumerate(dir_vec_pairs):
        dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
        dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length

    return dir_vec

def get_root_properties(actual_vertices):
    #root is index 0
    
    actual_vertices = actual_vertices.reshape(-1,18,3)
    
    root_positions = actual_vertices[:,0,:]
    
    future_points = root_positions[1:]
    current_points = root_positions[:-1]
    
    #pure vectors
    dir_vectors = future_points-current_points
    dir_vectors = np.vstack(([0,0,0],dir_vectors))
    return dir_vectors