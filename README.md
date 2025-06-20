This is a fork of [ESPNet](https://github.com/espnet/espnet) that intends to fine-tune [RawNet3](https://github.com/Jungjee/RawNet) model on our proprietary dataset. ESPNet provides complete pipelines for end-to-end speech processing on various tasks. We are only interested in the [speaker representation](https://espnet.github.io/espnet/recipe/spk1.html) part.

## Environment Installation

Follow the official [installation guide](https://espnet.github.io/espnet/installation.html) of ESPNet. You may need the latest clone from the official repository for the installation scripts to work.

## Changes We Made

According to the [HuggingFace page](https://huggingface.co/espnet/voxcelebs12_rawnet3), we checked-out [this commit](https://github.com/aik2mlj/espnet/commit/0c489a83607efb8e21331a9f01df21aac58c2a88) to ensure the compatibility with the released pre-trained RawNet3 model.

Our proprietary dataset consists of:

- [_hooktheory_](https://console.cloud.google.com/storage/browser/music-dataset-hooktheory-audio): 5.7G, 2725 singers, 8082 songs. We created a training configuration under `egs2/hooktheory/spk1` in this repository.

- [_metamidi_](https://console.cloud.google.com/storage/browser/metamidi-complete): 269G, 9558 singers, 57365 songs. The audio part of the [MetaMIDI](https://github.com/jeffreyjohnens/MetaMIDIDataset) dataset, using the singer & genre metadata it has. The configuration is under `egs2/metamidi/spk1`.

- [_12m_](https://console.cloud.google.com/storage/browser/12m-youtube): 2.3T, 59190 singers, 586791 songs. Partially scanned version of the [_LAION-DISCO-12M_](https://laion.ai/blog/laion-disco-12m/) dataset. The configuration is under `egs2/12m/spk1`.

The preparation of dataset mainly follows this paradigm (the order can change for some processes):
- Filtering out duplication of singers / songs, non-singing content if applicable.
- Going through [Demucs](https://github.com/facebookresearch/demucs) for singing voice separation.
- Downsampling to 16khz.
- Using `pydub.silence.split_on_silence` for singing activity detection, getting multiple singing segments for each song.
- Making `test_pairs.txt` for validation.

The datasets are in the same format as [VoxCeleb](https://mm.kaist.ac.kr/datasets/voxceleb/).
```
test
└── wav
    └── <singer_id>
        └── <song_name>
            ├── 00001.wav
            └── 00002.wav
dev
└── wav ...
```

For each dataset, see the `README.md` under `egs2/<dataset-name>` for details of its preparation.

## How to Train / Fine-tune

For each dataset configuration, you may want to change the parameters in `<base_path>/conf/train_rawnet3.yaml` (e.g., `batch_size`, `wandb` settings) before training. Please also take a look at `spk.sh` to understand what each step does in the pipeline. You may also specify the stage to start from by adding `--stage N` to the command, the last stage by adding `--stop_stage N`, and the stages to skip by adding `--skip_stages=A B`.
- During stage 1 (data preparation), the script will download [Musan](https://www.openslr.org/17/) and [RIR_NOISES](https://www.openslr.org/28/) for augmentation. (See `<base_path>/local/data.sh`). You may need to manually copy the `*.scp` files stored under `<base_path>` to `<base_path>/spk1/data/` in order to proceed in stage 4.
- We noticed that you may need to temporarily decrease the `num_workers` or `batch_size` for the stage 4 (stats collection) to finish. You can do them separately to change the parameters back.
  ```shell
  ./run.sh  --stop_stage 4  # run the pipeline up to stage 4
  ./run.sh --stage 5  # once successful, continue the pipeline on stage 5
  ```

For the proprietary dataset of

### [hooktheory](https://console.cloud.google.com/storage/browser/music-dataset-hooktheory-audio)

- Place the content under [this google cloud bucket](https://console.cloud.google.com/storage/browser/music-dataset-hooktheory-audio/for_espnet) under `egs2/hooktheory/`. Notice that you should have both the `hooktheory/` folder and the `test_pairs.txt` file.
- (Optional) Download the pre-trained RawNet3 checkpoint from [HuggingFace](https://huggingface.co/espnet/voxcelebs12_rawnet3) for fine-tuning.
  ```shell
  cd egs2/hooktheory/spk1
  mkdir pretrained && cd pretrained
  # clone the huggingface repo and put `40epoch.pth` here
  ```
- Run the following command to start the pipeline. Notice that for fine-tuning, the stored `loss.weight` values in the checkpoint are discarded since the last classification layer is different given a different number of speaker/singer classes.
  ```shell
  cd egs2/hooktheory/spk1
  ./run.sh  # for training from scratch
  ./run.sh --pretrained_model pretrained/40epoch.pth:::loss.weight  # for fine-tuning
  ```

### [metamidi](https://console.cloud.google.com/storage/browser/metamidi-complete)

Same as above, the base path is `egs2/metamidi/spk1`. The processed data is uploaded [here](https://console.cloud.google.com/storage/browser/metamidi-complete/for_espnet/), please place both the `metamidi/` folder and the `test_pairs.txt` file under `egs2/metamidi/`.

### [12m](https://console.cloud.google.com/storage/browser/12m-youtube)

Same as above, the base path is `egs2/12m/spk1`. The processed data is uploaded [here](https://console.cloud.google.com/storage/browser/12m-youtube/for_espnet/), please place both the `12m/` folder and the `test_pairs.txt` file under `egs2/12m/`.
