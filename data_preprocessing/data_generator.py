import glob
import librosa
import numpy as np
import os
import shutil
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize
import json
import pickle
from pre_processing_utils.body_utils import *
from pre_processing_utils.audio_utils import *
from tqdm import tqdm

def extract_data(data_type, duration, aist_plusplus_final_folder_path, wav_folder_path):
    seq_name = []
    audio_mfccs = []
    audio_chroma = []
    audio_sc = []
    audio_beats = []

    target_pose_vecs = []
    target_pose_vertices = []

    root_trans = []
    data_names = []

    with open('{}/splits/crossmodal_{}.txt'.format(aist_plusplus_final_folder_path, data_type), 'r') as f:
        for line in f:
            line = line.replace('\n','')
            data_names.append(line)

    
    with open('{}/ignore_list.txt'.format(aist_plusplus_final_folder_path), 'r') as f:
        for line in f:
            line = line.replace('\n','')
            if line in data_names:
                idx = data_names.index(line)
                data_names.pop(idx)

                print("dumped")

    data_paths = {}

    for name in data_names:
        audio_name = name.split('_')[-2]
        motion_path = '{}/keypoints3d/{}.pkl'.format(aist_plusplus_final_folder_path, name)
        audio_path = '{}/{}.wav'.format(wav_folder_path, audio_name)

        data_paths[name] = {
            'audio_name': audio_name,
            'motion_path': motion_path,
            'audio_path': audio_path
        }

    drop_counter = 0
    total = 0

    remove_list = []
    for key in tqdm(data_paths, total=len(list(data_paths.keys()))):

        pose_file = data_paths[key]['motion_path']

        with open(pose_file,'rb') as f:
            data = pickle.load(f)['keypoints3d_optim']

        if len(data) < 60 * duration:
            remove_list.append(key)
            drop_counter +=1

        total+=1

    print("Dropped {}/{}".format(drop_counter, total))


    for key in remove_list:
        data_paths.pop(key)

    for key in data_paths:

        seq_name_content = key
        audio_clip = data_paths[key]['audio_path']
        pose_file = data_paths[key]['motion_path']
       
        y,sr = librosa.load(audio_clip,sr=22500,duration=duration)

        mfcc_feat = get_mfcc_features(y,sr=22500,num_mfcc=14)
        chroma_feat = get_chroma_cens(y,sr=22500)
        sc_feat = get_sc(y,sr=22500)
        beats_feat = get_beats(y,sr=22500)

        target_actual_pose_seq = []
        target_actual_dir_vec_seq = []

        with open(pose_file,'rb') as f:
            data = pickle.load(f)['keypoints3d_optim']

        data = convert_aist_to_m2ad(data)

        for i in range(0,60*duration,6):
            pose = data[i]
            pose = np.array(pose)

            if len(pose.shape) > 0:

                pose = pose.reshape(1,18,3)
                vec = get_vec(pose,dir_vec_pairs)
                vec = vec.reshape(-1)
                target_actual_dir_vec_seq.append(vec)
                target_actual_pose_seq.append(data[i])

        if len(target_actual_dir_vec_seq) == 10 * duration:

            target_actual_pose_seq = np.array(target_actual_pose_seq)
            target_actual_dir_vec_seq = np.array(target_actual_dir_vec_seq)

            audio_mfccs.append(mfcc_feat)
            audio_chroma.append(chroma_feat)
            audio_sc.append(sc_feat)
            seq_name.append(seq_name_content)
            audio_beats.append(beats_feat)

            target_pose_vecs.append(target_actual_dir_vec_seq)
            target_pose_vertices.append(target_actual_pose_seq)

            root_prop1 = get_root_properties(np.array(target_actual_pose_seq))
            root_trans.append(root_prop1)



    audio_mfccs = np.array(audio_mfccs)
    audio_chroma = np.array(audio_chroma)
    audio_sc = np.array(audio_sc)
    seq_name = np.array(seq_name,dtype=object)
    audio_beats = np.array(audio_beats,dtype=object)

    target_pose_vecs = np.array(target_pose_vecs)
    target_pose_vertices = np.array(target_pose_vertices)

    root_trans = np.array(root_trans)

    with open('./Data/AIST_{}_data_with_root.pkl'.format(data_type),'wb') as f:

        final_data = {
            'audio_mfccs': audio_mfccs,
            'audio_chroma': audio_chroma,
            'audio_sc': audio_sc,
            'audio_beats': audio_beats,
            'seq_name': seq_name,
            'target_pose_vecs': target_pose_vecs,
            'target_pose_vertices': target_pose_vertices,
            'root_trans': root_trans
        }

        pickle.dump(final_data, f)


    print(audio_mfccs.shape)
    print(target_pose_vecs.shape)

if __name__ == "__main__":

    for data_type in ['train', 'test']:
        extract_data(data_type, 7)
        