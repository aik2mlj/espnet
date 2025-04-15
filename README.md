This is a fork of [ESPNet](https://github.com/espnet/espnet) that intends to fine-tune [RawNet3](https://github.com/Jungjee/RawNet) model on our proprietary dataset. ESPNet provides complete pipelines for end-to-end speech processing on various tasks. We are only interested in the [speaker representation](https://espnet.github.io/espnet/recipe/spk1.html) part.

## Environment Installation

Follow the official [installation guide](https://espnet.github.io/espnet/installation.html) of ESPNet. You may need the latest clone from the official repository for the installation scripts to work.

## Changes We Made

According to the [HuggingFace page](https://huggingface.co/espnet/voxcelebs12_rawnet3), we checked-out [this commit](https://github.com/aik2mlj/espnet/commit/0c489a83607efb8e21331a9f01df21aac58c2a88) to ensure the compatibility with the released pre-trained RawNet3 model.

Our proprietary dataset consists of:

- [_hooktheory_](https://console.cloud.google.com/storage/browser/music-dataset-hooktheory-audio): A downsampled arrangement of sample rate 16kHz is placed at the `for_espnet` in the google cloud bucket. We created a training configuration under `egs2/hooktheory/spk1` in this repository.

## How to Fine-tune

For the proprietary dataset of

### [hooktheory](https://console.cloud.google.com/storage/browser/music-dataset-hooktheory-audio)

- Place the content in `for_espnet/audio_16k` in the google cloud bucket under `egs2/hooktheory/hooktheory/`.
- You may want to change some parameters at `egs2/hooktheory/spk1/conf/train_rawnet3.yaml` (e.g., `batch_size`, `wandb` settings).
- Download the pre-trained RawNet3 checkpoint from [HuggingFace](https://huggingface.co/espnet/voxcelebs12_rawnet3)
  ```shell
  cd egs2/hooktheory/spk1
  mkdir pretrained && cd pretrained
  # clone the huggingface repo and put `40epoch.pth` here
  ```
- Run the following command to start the pipeline. Notice that the stored `loss.weight` values in the checkpoint are discarded since the last classification layer is different given a different number of speaker/singer classes.
  ```shell
  cd egs2/hooktheory/spk1
  ./run.sh --pretrained_model pretrained/40epoch.pth:::loss.weight
  ```
  - Please take a look at `spk.sh` to get a sense of each stage in the pipeline. You may also specify the stage to start from by adding `--stage N` to the command, the last stage by adding `--stop_stage N`, and the stages to skip by adding `--skip_stages=A B`.
  - During stage 1 (data preparation), the script will download [Musan](https://www.openslr.org/17/) and [RIR_NOISES](https://www.openslr.org/28/) for augmentation. (See `egs2/hooktheory/spk1/local/data.sh`). You may need to manually copy the `*.scp` files stored under `egs2/hooktheory/` to `egs2/hooktheory/spk1/data/` in order to proceed in stage 4.
- Alternatively, you may want to train the model from scratch. In this case, you can remove the `--pretrained_model` argument from the command above.

### MetaMIDI

Basically the same as the above, but you need to change the base path to `egs2/metamidi/spk1`. The processed data haven't been uploaded to google cloud bucket yet.
