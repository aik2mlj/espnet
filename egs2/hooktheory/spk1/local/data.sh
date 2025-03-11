#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=3
stop_stage=100
n_proc=8

data_dir_prefix= # root dir to save datasets.

trg_dir=data

. utils/parse_options.sh
. db.sh
. path.sh
. cmd.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z ${data_dir_prefix} ]; then
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/hooktheory"
    data_dir_prefix=${MAIN_ROOT}/egs2/hooktheory
else
    log "Root dir set to ${HOOKTHEORY}"
    data_dir_prefix=${HOOKTHEORY}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Download Musan and RIR_NOISES for augmentation."

    if [ ! -f ${data_dir_prefix}/rirs_noises.zip ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/28/rirs_noises.zip
    else
        log "RIRS_NOISES exists. Skip download."
    fi

    if [ ! -f ${data_dir_prefix}/musan.tar.gz ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/17/musan.tar.gz
    else
        log "Musan exists. Skip download."
    fi

    if [ -d ${data_dir_prefix}/RIRS_NOISES ]; then
        log "Skip extracting RIRS_NOISES"
    else
        log "Extracting RIR augmentation data."
        unzip -q ${data_dir_prefix}/rirs_noises.zip -d ${data_dir_prefix}
    fi

    if [ -d ${data_dir_prefix}/musan ]; then
        log "Skip extracting Musan"
    else
        log "Extracting Musan noise augmentation data."
        tar -zxvf ${data_dir_prefix}/musan.tar.gz -C ${data_dir_prefix}
    fi

    # make scp files
    for x in music noise speech; do
        find ${data_dir_prefix}/musan/${x} -iname "*.wav" >${data_dir_prefix}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" >${data_dir_prefix}/rirs.scp
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >>${data_dir_prefix}/rirs.scp
    log "Stage 3, DONE."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4, Change into kaldi-style feature."
    mkdir -p ${trg_dir}/hooktheory_test
    mkdir -p ${trg_dir}/hooktheory_dev
    python local/data_prep.py --src "${data_dir_prefix}/hooktheory/test" --dst "${trg_dir}/hooktheory_test"
    python local/data_prep.py --src "${data_dir_prefix}/hooktheory/dev" --dst "${trg_dir}/hooktheory_dev"

    for f in wav.scp utt2spk spk2utt; do
        sort ${trg_dir}/hooktheory_test/${f} -o ${trg_dir}/hooktheory_test/${f}
        sort ${trg_dir}/hooktheory_dev/${f} -o ${trg_dir}/hooktheory_dev/${f}
    done

    # make test trial compatible with ESPnet.
    # TODO: veri_test2.txt should be provided by siqi
    python local/convert_trial.py --trial ${data_dir_prefix}/veri_test2.txt --scp ${trg_dir}/hooktheory_test/wav.scp --out ${trg_dir}/hooktheory_test

    log "Stage 4, DONE."

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
