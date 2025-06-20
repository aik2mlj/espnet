5.7G, 2725 singers, 8082 songs.

All the scripts are located under `spk1/local/`.

- Downsample to 16kHz (`gs://music-dataset-hooktheory-audio/cartesia-dataset/dec_10th/hooktheory_18k_melody_cartesia_44k_outputs` ⇒ `audio-16k/`)
    - `downsample_redirect.py`
- Singer ID mapping
    - `singer_id_mapper.py`
- Train-test split
    - `train_test_split_singer.py`
