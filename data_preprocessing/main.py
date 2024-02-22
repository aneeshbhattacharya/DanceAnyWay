from argparse import ArgumentParser
from create_sliced_dataset import create_dataset
from data_generator_train_extended import extract_data as extract_sliced_data_train
from data_generator import extract_data as extract_data_sm

def process_data(aist_plusplus_final_folder_path, wav_folder_path):

    train_list_path = '{}/splits/crossmodal_train.txt'.format(aist_plusplus_final_folder_path)
    filter_list_path = '{}/ignore_list.txt'.format(aist_plusplus_final_folder_path)

    training_list = []
    filter_list = []

    with open(train_list_path,'r') as file:
        for line in file:
            training_list.append(line.rstrip()) 

    with open(filter_list_path,'r') as file:
        for line in file:
            filter_list.append(line.rstrip()) 
            
    # Created the sliced dataset
    create_dataset(training_list, filter_list, aist_plusplus_final_folder_path, wav_folder_path)

    # Create extended training pkl
    extract_sliced_data_train('train',7,aist_plusplus_final_folder_path, wav_folder_path)

    # Create normal testing scripts
    for data_type in ['train', 'test']:
        extract_data_sm(data_type,7,aist_plusplus_final_folder_path, wav_folder_path)

if __name__ == "__main__":
    
    parser = ArgumentParser(description='')

    parser.add_argument('--aist_plusplus_final_folder_path', '-a', help='Path to aist_plusplus_final_folder', type=str, default='./Data/aist_plusplus_final')
    parser.add_argument('--wav_folder_path', '-w', help='Path to wav folder', type=str, default='./Data/wav')
    args = parser.parse_args()

    aist_plusplus_final_folder_path = args.aist_plusplus_final_folder_path
    wav_folder_path = args.wav_folder_path
    
    train_list_path = '{}/splits/crossmodal_train.txt'.format(aist_plusplus_final_folder_path)
    filter_list_path = '{}/ignore_list.txt'.format(aist_plusplus_final_folder_path)

    training_list = []
    filter_list = []

    with open(train_list_path,'r') as file:
        for line in file:
            training_list.append(line.rstrip()) 

    with open(filter_list_path,'r') as file:
        for line in file:
            filter_list.append(line.rstrip()) 
            
    # Created the sliced dataset
    create_dataset(training_list, filter_list, aist_plusplus_final_folder_path, wav_folder_path)
    
    # Create extended training pkl
    extract_sliced_data_train('train',7,aist_plusplus_final_folder_path, wav_folder_path)
    
    # Create normal testing scripts
    for data_type in ['train', 'test']:
        extract_data_sm(data_type,7,aist_plusplus_final_folder_path, wav_folder_path)

    
    
