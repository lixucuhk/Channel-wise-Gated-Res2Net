#!/bin/bash
# This script extracts Spec, CQT, LFCC features for:
# ASVspoof2019 PA train, PA dev, PA eval, LA train, LA dev, LA eval

. ./cmd.sh
# . ./path.sh
set -e
specdir=`pwd`/raw_feats/spec
datadir=/scratch/xli/Data_Source/ASVspoof2019

stage=0

. ./parse_options.sh || exit 1

mkdir -p data/LA_train data/LA_dev data/LA_eval || exit 1

if [ $stage -le 0 ]; then
   echo "Stage 0: prepare dataset."
   for access_type in LA;do
       protofile=$datadir/${access_type}/ASVspoof2019_${access_type}_cm_protocols/ASVspoof2019.${access_type}.cm.train.trn.txt
       awk '{print $2" "$4}' $protofile >data/${access_type}_train/utt2systemID || exit 1
       awk '{print $2" "$1}' $protofile >data/${access_type}_train/utt2spk || exit 1
       feats_extraction/utt2spk_to_spk2utt.pl data/${access_type}_train/utt2spk >data/${access_type}_train/spk2utt || exit 1
       awk -v dir="${datadir}/${access_type}" -v type="${access_type}_train" '{print $2" sox "dir"/ASVspoof2019_"type"/flac/"$2".flac -t wav - |"}' $protofile >data/${access_type}_train/wav.scp || exit 1
       
       for dataset in dev eval;do
           protofile=$datadir/${access_type}/ASVspoof2019_${access_type}_cm_protocols/ASVspoof2019.${access_type}.cm.${dataset}.trl.txt
           awk '{print $2" "$4}' $protofile >data/${access_type}_${dataset}/utt2systemID || exit 1
           awk '{print $2" "$1}' $protofile >data/${access_type}_${dataset}/utt2spk || exit 1
           feats_extraction/utt2spk_to_spk2utt.pl data/${access_type}_${dataset}/utt2spk >data/${access_type}_${dataset}/spk2utt || exit 1
           awk -v dir="${datadir}/${access_type}" -v type="${access_type}_${dataset}" '{print $2" sox "dir"/ASVspoof2019_"type"/flac/"$2".flac -t wav - |"}' $protofile >data/${access_type}_${dataset}/wav.scp || exit 1
       done
    done
    echo "dataset finished"
    exit 0
fi

if [ $stage -le 1 ]; then
   echo "Stage 1: extract Spec feats."
   mkdir -p data/spec || exit 1
   for name in LA_train LA_dev LA_eval; do
       [ -d data/spec/${name} ] || cp -r data/${name} data/spec/${name} || exit 1
       feats_extraction/make_spectrogram.sh --feats-config conf/feats/spec.conf --nj 80 --cmd "$train_cmd" \
             data/spec/${name} exp/make_spec $specdir || exit 1
   done
   echo "Spec feats done"
   exit 0
fi

if [ $stage -le 2 ]; then
   echo "Stage 2: extract CQT feats."
   python3 feats_extraction/compute_feats.py --out_dir data/cqt --access_type LA --param_json_path conf/feats/cqt_48bpo_fmin15.json --num_workers 60 || exit 1
   python3 feats_extraction/GenKaldiFeats.py --access_type LA --work_dir data/cqt || exit 1

   # uncomment below for removing numpy data to save space.
   #for name in LA_train LA_dev LA_eval; do
   #    rm -rf data/cqt/${name}/*.npy || exit 1
   #done
   exit 0
fi

if [ $stage -le 3 ]; then
   echo "Stage 3: extract LFCC feats."
   python3 feats_extraction/compute_feats.py --out_dir data/lfcc --access_type LA --param_json_path conf/feats/lfcc.json --num_workers 60 --feature lfcc || exit 1
   python3 feats_extraction/GenKaldiFeats.py --access_type LA --work_dir data/lfcc || exit 1

   # uncomment below for removing numpy data to save space.
   #for name in LA_train LA_dev LA_eval; do
   #    rm -rf data/lfcc/${name}/*.npy || exit 1
   #done
   exit 0
fi

if [ $stage -le 4 ]; then
   echo "Stage 4: truncate features and generate labels."
   for name in LA_train LA_dev LA_eval; do
       for feat_type in spec cqt lfcc; do
           echo "Processing $name $feat_type"
           [ -f data/${feat_type}/${name}/utt2systemID ] || cp data/${name}/utt2systemID data/${feat_type}/${name}/utt2systemID || exit 1
           python3 feats_extraction/feat_slicing.py --in-scp data/${feat_type}/${name}/feats.scp --out-scp data/${feat_type}/${name}/feats_slicing.scp --out-ark data/${feat_type}/${name}/feats_slicing.ark || exit 1
           python3 feats_extraction/convertID2index.py --scp-file data/${feat_type}/${name}/feats_slicing.scp --sysID-file data/${feat_type}/${name}/utt2systemID --out-file data/${feat_type}/${name}/utt2index --access-type ${name:0:2} || exit 1
       done
   done
   exit 0
fi

