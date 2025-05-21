#!/usr/bin/env bash
set -e
set -u
set -o pipefail

spk_config=conf/train_rawnet3.yaml

train_set="12m_dev"
valid_set="12m_test"
# cohort_set="voxceleb2_test"
# can have multiple test sets, e.g. "test1 test2"
test_sets="12m_test"
feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    "$@" # --cohort_set ${cohort_set} \
