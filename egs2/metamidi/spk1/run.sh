#!/usr/bin/env bash
set -e
set -u
set -o pipefail

spk_config=conf/train_rawnet3.yaml

train_set="metamidi_dev"
valid_set="metamidi_test"
# cohort_set="voxceleb2_test"
test_sets="metamidi_test"
feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    "$@" # --cohort_set ${cohort_set} \
