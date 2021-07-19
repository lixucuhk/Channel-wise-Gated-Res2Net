import json
import codecs
import os, sys
import numpy as np 
import argparse
from multiprocessing import Process

from feats_extraction.utt2feats import logpowcqt, extract_lfcc, extract_stft

def build_from_path(wavlist, out_dir, feature, feats_conf):
    num_done = 0
    for wav_path in wavlist:
        utt_idx = os.path.basename(wav_path).rstrip('.flac')
        process_utterance(out_dir, utt_idx, wav_path, feature, feats_conf)
        num_done += 1
        if num_done % 50 == 0:
            print('Done %d utts.' %(num_done))

def process_utterance(out_dir, utt_idx, wav_path, feature, feats_conf):
    if feature == 'cqt':
        feats = logpowcqt(wav_path, feats_conf)
    elif feature == 'lfcc':
        feats = extract_lfcc(wav_path, feats_conf)
    elif feature == 'stft':
        feats = extract_stft(wav_path, feats_conf)
    feats_filename = os.path.join(out_dir, utt_idx+".npy")
    np.save(feats_filename, feats.astype(np.float32), allow_pickle=False)


def preprocess(wavlist, out_dir, feature, feats_conf, nj):
    os.makedirs(out_dir, exist_ok=True)
    wavlist_nj = np.array_split(wavlist, nj)
    process_list = []
    nj_index = 0
    for wavlist_ in wavlist_nj:
        p = Process(target=build_from_path, args=(wavlist_, out_dir, feature, feats_conf))
        p.start()
        process_list.append(p)
        print('job %d started.' %(nj_index))
        nj_index += 1

    for p in process_list:
        p.join()

def GenWavList(WavScpFile):
    with open(WavScpFile, 'r') as rf:
         data = []
         for line in rf.readlines():
             data.append(line.split()[2])

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--out_dir', type=str, default='data/cqt')
    parser.add_argument('--access_type', type=str, default='LA')
    parser.add_argument('--param_json_path', type=str, default='./conf/cqt.json')
    parser.add_argument('--feature', type=str, default='cqt') ## `cqt` or `lfcc` or `stft`
    args = parser.parse_args()

    num_workers = args.num_workers
    print("number of workers: ", num_workers)

    feature = args.feature
    print('Extracting %s...' %(feature))

    wavfile_list_train = GenWavList('data/%s_train/wav.scp' %(args.access_type))
    wavfile_list_dev   = GenWavList('data/%s_dev/wav.scp' %(args.access_type))
    wavfile_list_eval  = GenWavList('data/%s_eval/wav.scp' %(args.access_type))

    feats_conf = args.param_json_path
    with open(feats_conf) as json_file:
        print(json.load(json_file))

    print("Preprocess training data ...")
    preprocess(wavfile_list_train, args.out_dir+'/'+args.access_type+'_train', feature, feats_conf, num_workers)

    print("Preprocess dev data ...")
    preprocess(wavfile_list_dev, args.out_dir+'/'+args.access_type+'_dev', feature, feats_conf, num_workers)

    print("Preprocess eval data ...")
    preprocess(wavfile_list_eval, args.out_dir+'/'+args.access_type+'_eval', feature, feats_conf, num_workers)

    print("DONE!")
    sys.exit(0)



