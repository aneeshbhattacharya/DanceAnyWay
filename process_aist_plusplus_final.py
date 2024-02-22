from argparse import ArgumentParser
from data_preprocessing.main import process_data

if __name__ == "__main__":
    
    parser = ArgumentParser(description='')

    parser.add_argument('--aist_plusplus_final_folder_path', '-a', help='Path to aist_plusplus_final_folder', type=str, default='./Data/aist_plusplus_final')
    parser.add_argument('--wav_folder_path', '-w', help='Path to wav folder', type=str, default='./Data/wav')
    args = parser.parse_args()

    aist_plusplus_final_folder_path = args.aist_plusplus_final_folder_path
    wav_folder_path = args.wav_folder_path
    
    print(aist_plusplus_final_folder_path)
    
    process_data(aist_plusplus_final_folder_path, wav_folder_path)