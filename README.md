# DanceAnyWay: Synthesizing Beat-Guided 3D Dances With Randomized Temporal Contrastive Learning

This is the readme to use the official code for our [AAAI 2024](https://aaai.org/aaai-conference/) paper "DanceAnyWay: Synthesizing Beat-Guided 3D Dances With Randomized Temporal Contrastive Learning". You can find the paper draft [here](https://arxiv.org/abs/2303.03870). Camera-ready version coming soon!

[![](https://markdown-videos-api.jorgenkh.no/youtube/SRMI2HzIuLs)](https://youtu.be/SRMI2HzIuLs)  

## Pre-Requisites
Our scripts have been tested on Ubuntu 20.04 LTS with
- Python 3.7
- Cuda 11.6

## Installation
1. Clone this repository.

2. [Optional but recommended] Create a conda environment for the project and activate it:

```
conda create daw-env python=3.7
conda activate daw-env
```

4. Install PyTorch following the [official instructions](https://pytorch.org/).

5. Install all other package requirements:

```
pip install -r requirements.txt
```
Note: You might need to manually uninstall and reinstall `numpy` for `torch` to work. You might need to manually uninstall and reinstall `matplotlib` and `kiwisolver` for them to work.


## Downloading the Dataset
1. Download AIST++ dataset [here](https://google.github.io/aistplusplus_dataset/download.html) and store it in `data_preprocessing/Data`.

2. Run the following command in the data_preprocessing directory:
```
python process_aist_plusplus_final.py
```
Note: This data extraction may take a few hours depending on the configuration of the device.

3. Download the presequences file [here](Google drive link).


## Pre-Trained Models
We provide the pretrained models [here](Google drive link). Save these models inside their respective `train_results` directories as `checkpoint.pt`.


## Running the Code
1. In order to train the models, run the following command in each of the respective directories:
```
python train.py
```
Note: In order to change the parameters of the model or training, please modify the config file provided for each model.

2. In order to test the models on the test dataset, run the following command in each of the respective directories:
```
python test.py
```
Note: Results will be generated in a new directory named `test_results` by default. This behavior may be changed as a command line argument.

3. In order to evaluate the model on in-the-wild music, run the following command in the `dance_generator` directory:
```
python evaluate.py --music_file music.wav
```
Note: By default, the model will generate a dance for the first 7 seconds. For continuous generation, please run the following command:
```
python evaluate.py --music_file music.wav --infinite_gen True
```

# Citation
Please use the following citation if you find our work useful:
```
@inproceedings{bhattacharya2024danceanyway,
author = {Bhattacharya, Aneesh and Paranjape, Manas and Bhattacharya, Uttaran and Bera, Aniket},
title = {DanceAnyWay: Synthesizing Beat-Guided 3D Dances With Randomized Temporal Contrastive Learning},
year = {2024},
publisher = {Association for the Advancement of Artificial Intelligence},
address = {New York, NY, USA},
booktitle = {Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence},
series = {AAAI '24}
}
```
