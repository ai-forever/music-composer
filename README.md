# Music Composer
This repository is dedicated to synthesizing symbolic music in MIDI format using the Music Transformer model. In the repository, you can find a demo laptop for generating on a GPU Google Colab instance, data preparation and model training code.

## Table of Contents
1. [Demo notebook] (# demo-laptop)
2. [Model code] (# model code)
3. [Data] (# data)
4. [Training] (# training)


## Demo notebook

Jupyter Notebook can be opened on Colab by clicking on the button:

[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/music-composer/blob/main/src/Music_Composer_Demo_Colab.ipynb)

It unrolls the environment, loads the code and weights for synthesis. Generation parameters are set in the generation control panel, and you can listen and download the results in the last cell.

❗ Make sure the GPU instance is being used at startup. It is possible to synthesize on a CPU, but it takes significantly more time.

## Model code
Located in [folder](https://github.com/sberbank-ai/music-composer/tree/main/src/lib/model). 
Consists of three main parts:
- Positional encoding - normal positional encoding for transformer models
- Relative Positional Representation - a module with the implementation of Relative Attention
- Transformer - the model itself is a transformer 

Model code and relative attention taken from [repository](https://github.com/gwinndr/MusicTransformer-Pytorch).

## Data
To demonstrate the encoding script, we provide several MIDI files from our training sample. They are located in the src / test_dataset folder and are divided into folders by genre. Each folder contains one file to check. You can start preparing event-based versions of these files using the command:
```python encode_dataset.py```

The folder with the source MIDI and the folder for the results are set inside the script through the variables `DATA_DIR`,` OUTPUT_DIR`. Dataset files with file paths will be created in `DS_FILE_PATH`. The genre list is specified using `GENRES`, and the maximum record length in event tokens is` MAX_LEN`.

For demonstration, we also provide the output of this command in the encoded_dataset folder. It contains tensors with MIDI converted to event-based format. They can be loaded using the standard `torch.load (file_path)`
Datasets can be used as public MIDI for training:
[MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
[Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)
[GiantMIDI-Piano Dataset](https://github.com/bytedance/GiantMIDI-Piano)

There is another way to get MIDI files - transcribing wave files with music. An approach like [Onset-frames] (https://magenta.tensorflow.org/onsets-frames) can help with this.
As music for transcription, you can use for example [Free Music Archive] (https://github.com/mdeff/fma).
❗Significant resources may be required for transcribing, but this is exactly what will allow to get around the main limitation of the current models of symbolic music generation - the absence of large corpora with notes.
❗ After transcription, it is recommended to analyze the results and filter out bad recordings.
## Training
A script for training a model on prepared data can be run using:
```python train.py```
Training parameters are set inside the script in the params variable. A description of each of the parameters will be given later in this section.
