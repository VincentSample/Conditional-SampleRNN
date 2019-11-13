# Conditional-SampleRNN

PyTorch implementation of the Conditional SampleRNN, 

## Dependencies
It's highly recommmended to create a conda environment. Repository tested with :

- Python 3.7.4
- CUDA Version 8.0.61
- CuDNN 5110

- matplotlib 3.1.0
- natsort 6.0.0
- numpy 1.16.4
- librosa 0.7.0
- torch 0.4.1.post2

To run the scripts, you need
- ffmpeg 4.2
- youtube-dl 2019.8.13 

## Datasets
Either download your own dataset and preprocess it in the the next step or download the ambient dataset with `datasets/download-ambient.sh`

## Preprocessing Data.
The script `datasets/preprocess.sh` preprocesses the data: the last step before training. The input dataset must be as follows inside the `datasets` folder. It's tested with .mp3 files but should work with any audio file, stereo or mono, no matter the sample rate or format. If it does not work, convert it first to .mp3

```
name_of_dataset
├── class_1
│   ├── file1.mp3
│   └── file2.mp3
├── class_2
├── class_3
└── class_4
```
Then, make sure that the script is executable `chmod +x preprocess.sh` and  run `./preprocess.sh name_of_dataset 8 name_of_dataset_parts`
At the **end** of the line, two options are available :
```
[-r] : remove the input files (mp3 files) at the end of preprocessing.
[-b] : balance classes. Figure out which class has less data and truncate the other classes.
```
The balance option is highly recommended to optimise training.
If you're doing it with the ambient dataset, run:
`./preprocess.sh ambient 8 ambient_parts -r -b`

A file `map_file.txt` maps each wav file to its source mp3 file (useful if additional information needs to be stored)

A file `map_class.txt` maps each class to its name (sad<->2)

## Training 

run `python train.py -h`
```
usage: train.py [-h] --exp EXP --frame_sizes FRAME_SIZES [FRAME_SIZES ...]
                --dataset DATASET [--n_rnn N_RNN] [--dim DIM]
                [--learn_h0 LEARN_H0] [--q_levels Q_LEVELS]
                [--seq_len SEQ_LEN] [--weight_norm WEIGHT_NORM]
                [--batch_size BATCH_SIZE] [--val_frac VAL_FRAC]
                [--test_frac TEST_FRAC]
                [--keep_old_checkpoints KEEP_OLD_CHECKPOINTS]
                [--datasets_path DATASETS_PATH] [--results_path RESULTS_PATH]
                [--epoch_limit EPOCH_LIMIT] [--resume RESUME]
                [--sample_rate SAMPLE_RATE] [--n_samples N_SAMPLES]
                [--sample_length SAMPLE_LENGTH]
                [--loss_smoothing LOSS_SMOOTHING] [--cuda CUDA]
                [--nb_classes NB_CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  --exp EXP             experiment name
  --frame_sizes FRAME_SIZES [FRAME_SIZES ...]
                        frame sizes in terms of the number of lower tier
                        frames, starting from the lowest RNN tier
  --dataset DATASET     dataset name - name of a directory in the datasets
                        path (settable by --datasets_path)
  --n_rnn N_RNN         number of RNN layers in each tier (default: 1)
  --dim DIM             number of neurons in every RNN and MLP layer (default:
                        1024)
  --learn_h0 LEARN_H0   whether to learn the initial states of RNNs (default:
                        True)
  --q_levels Q_LEVELS   number of bins in quantization of audio samples
                        (default: 256)
  --seq_len SEQ_LEN     how many samples to include in each truncated BPTT
                        pass (default: 1024)
  --weight_norm WEIGHT_NORM
                        whether to use weight normalization (default: True)
  --batch_size BATCH_SIZE
                        batch size (default: 128)
  --val_frac VAL_FRAC   fraction of data to go into the validation set
                        (default: 0.1)
  --test_frac TEST_FRAC
                        fraction of data to go into the test set (default:
                        0.1)
  --keep_old_checkpoints KEEP_OLD_CHECKPOINTS
                        whether to keep checkpoints from past epochs (default:
                        False)
  --datasets_path DATASETS_PATH
                        path to the directory containing datasets (default:
                        datasets)
  --results_path RESULTS_PATH
                        path to the directory to save the results to (default:
                        results)
  --epoch_limit EPOCH_LIMIT
                        how many epochs to run (default: 1000)
  --resume RESUME       whether to resume training from the last checkpoint
                        (default: True)
  --sample_rate SAMPLE_RATE
                        sample rate of the training data and generated sound
                        (default: 16000)
  --n_samples N_SAMPLES
                        number of samples to generate in each epoch (default:
                        1)
  --sample_length SAMPLE_LENGTH
                        length of each generated sample (in samples) (default:
                        80000)
  --loss_smoothing LOSS_SMOOTHING
                        smoothing parameter of the exponential moving average
                        over training loss, used in the log and in the loss
                        plot (default: 0.99)
  --cuda CUDA           whether to use CUDA (default: True)
  --nb_classes NB_CLASSES
                        number of classes (labels) of training data
```

Remember, with `frame_sizes`, you choose the upsampling ratio. If you choose 16, 8, 4 , it's a *4-tier module* with 
- sample_level_frame_size = 16
- bottom_rrn_module_frame_size = 16
- medium_rrn_module_frame_size = 16\*8=112
- top_rrn_module_frame_size = 16\*8\*4=448

Now here's the command used for this dataset :
`python train.py --exp ambient1 --frame_sizes 16 4 --n_rnn 2 --dataset ambient_parts --n_samples 5 --nb_classes 4`

## Generate audio

At each end of epoch `n_samples * nb_classes` samples are created, the first n_samples belong to class 0, the n_samples following to class 1 etc.

You can also generate custom audio files with `generate_audio.py`
If you run without the `--cond` argument, it's going to generate samples juste like at the end of each epoch.
If you run it with the `--cond` argument, it's going to generate samples with the argument one_hot vector.

For example, running 

`python generate_audio.py --exp creepyspace --frame_sizes 16 4 --nb_classes 4 --generate_from path_to_model --n_samples 40 --sample_length 160000 --generate_to path_to_results_folder --cond 0 0 1 1`

will produce 40 samples of 10 seconds each, with every sample generated with the pseudo-one-hot encoding vector `[0, 0, 1, 1]`

## Questions

Feel free to ask any questions !
