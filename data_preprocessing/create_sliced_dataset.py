import argparse
import os
import glob
import pickle
import librosa
import soundfile as sf
from tqdm import tqdm

def get_clean_list(data_list, filter_list):
    clean_list = []
    for data_point in data_list:
        if data_point not in filter_list:
            clean_list.append(data_point)
        else:
            print("Dropped {}".format(data_point))

    print("Num samples: {}".format(len(clean_list)))
    return clean_list

def slice_motion_and_audio_KPTS(data_path, data_type, aist_plusplus_final_folder_path, wav_folder_path, sampling_length=7, frame_rate = 60, overlap = 0.5):

    audio_step_size = int(0.5 * 44100)

    motion_step_size = int(overlap * frame_rate)
    sliced_audio_save_path = './Data/Sliced_Dataset/{}/audios'.format(data_type)
    sliced_motion_save_path = './Data/Sliced_Dataset/{}/keypoints'.format(data_type)

    if not os.path.exists(sliced_motion_save_path):
        os.makedirs(sliced_motion_save_path)
    if not os.path.exists(sliced_audio_save_path):
        os.makedirs(sliced_audio_save_path)

    basename = os.path.basename(data_path).split('.')[0]

    data_path = '{}/keypoints3d/{}.pkl'.format(aist_plusplus_final_folder_path, basename)

    audio_name = basename.split('_')[-2]
    audio_name = audio_name.replace('cAll','c01')
    audio_data_path = '{}/{}.wav'.format(wav_folder_path, audio_name)
    audio, sr = librosa.load(audio_data_path, sr=44100)

    with open(data_path, 'rb') as file:
        data = pickle.load(file)


    motion_poses = data['keypoints3d_optim']

    motion_duration = len(motion_poses)
    motion_curr_window_start = 0
    motion_curr_window_end = motion_curr_window_start + (sampling_length * 60)

    audio_curr_window_start = 0
    audio_curr_window_end = audio_curr_window_start + (sampling_length * sr)

    slice_count = 0

    # print("Processing {}".format(basename))

    while(motion_curr_window_end<=motion_duration or slice_count == 0):

        if motion_curr_window_end > motion_duration:
            motion_curr_window_end = int((motion_duration//60)) * 60
            audio_curr_window_end = int(motion_curr_window_end//60) * 44100

        # print('Motion slice from {} to {} i.e {}-{}s'.format(motion_curr_window_start, motion_curr_window_end, motion_curr_window_start/60, motion_curr_window_end/60))

        sub_sequence_poses = motion_poses[motion_curr_window_start:motion_curr_window_end]
        
        # print('Motion len {}'.format(len(sub_sequence_poses)/60))
        
        motion_curr_window_start += motion_step_size
        motion_curr_window_end = motion_curr_window_start + (sampling_length * 60)

        smpl_data = {
            'keypoints3d_optim': sub_sequence_poses,
        }

        save_name = '{}/{}_slice_{}.pkl'.format(sliced_motion_save_path, basename, slice_count)
        with open(save_name,'wb') as f:
            pickle.dump(smpl_data, f)


        # print('Audio slice from {} to {}'.format(audio_curr_window_start/sr, audio_curr_window_end/sr))

        sliced_audio = audio[audio_curr_window_start:audio_curr_window_end]

        # print("Audio len {}".format(len(sliced_audio)/44100))

        audio_curr_window_start += audio_step_size
        audio_curr_window_end = audio_curr_window_start + (sampling_length * sr)

        save_name = '{}/{}_slice_{}.wav'.format(sliced_audio_save_path, basename, slice_count)
        sf.write(save_name, sliced_audio, samplerate=sr)
        slice_count+=1

def slice_data(all_paths, aist_plusplus_final_folder_path, wav_folder_path):
    data_type='training'
    for path in tqdm(all_paths, total=len(all_paths)):
        motion_path = '{}/motions/{}.pkl'.format(aist_plusplus_final_folder_path, path)
        slice_motion_and_audio_KPTS(motion_path, data_type, aist_plusplus_final_folder_path, wav_folder_path)

def create_dataset(training_list, filter_list, aist_plusplus_final_folder_path, wav_folder_path):
    training_list = get_clean_list(training_list, filter_list)
    slice_data(training_list, aist_plusplus_final_folder_path, wav_folder_path)

if __name__ == "__main__":

    train_list_path = './Data/aist_plusplus_final/splits/crossmodal_train.txt'
    filter_list_path = './Data/aist_plusplus_final/ignore_list.txt'

    training_list = []
    filter_list = []

    with open(train_list_path,'r') as file:
        for line in file:
            training_list.append(line.rstrip()) 

    with open(filter_list_path,'r') as file:
        for line in file:
            filter_list.append(line.rstrip()) 
            
    create_dataset(training_list, filter_list)
