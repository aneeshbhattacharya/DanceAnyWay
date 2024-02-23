# DanceAnyWay: Synthesizing Beat-Guided 3D Dances With Randomized Temporal Contrastive Learning

This is the readme to use the official code for our [AAAI 2024](https://aaai.org/aaai-conference/) paper "DanceAnyWay: Synthesizing Beat-Guided 3D Dances With Randomized Temporal Contrastive Learning". You can find the paper [here](Arxiv link to paper)
Code and models coming soon!


## Installation
Our scripts have been tested on Ubuntu 18.04 LTS with
- Python 3.10
- Cuda 12.1

1. Clone this repository.

2. [Optional but recommended] Create a conda envrionment for the project and activate it.

```
conda create daw-env python=3.10
conda activate daw-env
```

4. Install PyTorch following the [official instructions](https://pytorch.org/).

5. Install all other package requirements.

```
pip install -r requirements.txt
```
Note: You might need to manually uninstall and reinstall `numpy` for `torch` to work. You might need to manually uninstall and reinstall `matplotlib` and `kiwisolver` for them to work.


## Downloading the dataset
1. Download AIST++ dataset [here](https://google.github.io/aistplusplus_dataset/download.html) and store it in data_preprocessing/Data

2. Run the following command in the data_preprocessing directory:
```
python process_aist_plusplus_final.py
```
Note: This data extraction may take a few hours depending on the configuration of the device.

3. Download the presequences file [here](Google drive link)


## Pre-trained models
We provide a the pretrained models [here](Google drive link)
Save these models inside their respective train_results directories as *checkpoint.pt*


## Running the code
1. In order to train the models, run the following command in each of the respective directories
```
python train.py
```
Note: In order to change the parameters of the model or training, please modify the config file provided for each model

2. In order to test the models on the test dataset, run the following command in each of the respective directories
```
python test.py
```
Note: Results will be generated in a new directory named *test_results* by default. This behavior may be changed as a command line argument

3. In order to evaluate the model on in-the-wild music, run the following command in the *dance_generator* directory
```
python evaluate.py --music_file music.wav
```
Note: By default, the model will generate a dance for the first 7 seconds. For infinite generation, please run the following command:
```
python evaluate.py --music_file music.wav --infinite_gen True
```

<!-- Please use the following citation if you find our work useful: -->
<!-- ```
@inproceedings{bhattacharya2021speech2affectivegestures,
author = {Bhattacharya, Uttaran and Childs, Elizabeth and Rewkowski, Nicholas and Manocha, Dinesh},
title = {Speech2AffectiveGestures: Synthesizing Co-Speech Gestures with Generative Adversarial Affective Expression Learning},
year = {2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
series = {MM '21}
}
```-->
