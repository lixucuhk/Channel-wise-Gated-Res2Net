import os


la_spec = {
        'train_scp': 'data/spec/LA_train/feats_slicing.scp',
        'train_utt2index': 'data/spec/LA_train/utt2index',
        'dev_scp': 'data/spec/LA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/spec/LA_dev/utt2index',
        'dev_utt2systemID': 'data/spec/LA_dev/utt2systemID',
        'scoring_dir': 'scoring/la_cm_scores/',
        'eval_scp': 'data/spec/LA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/spec/LA_eval/utt2index',
        'eval_utt2systemID': 'data/spec/LA_eval/utt2systemID',
        'isstft': False,
}

la_lfcc = {
        'train_scp': 'data/lfcc/LA_train/feats_slicing.scp',
        'train_utt2index': 'data/lfcc/LA_train/utt2index',
        'dev_scp': 'data/lfcc/LA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/lfcc/LA_dev/utt2index',
        'dev_utt2systemID': 'data/lfcc/LA_dev/utt2systemID',
        'scoring_dir': 'scoring/la_cm_scores/',
        'eval_scp': 'data/lfcc/LA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/lfcc/LA_eval/utt2index',
        'eval_utt2systemID': 'data/lfcc/LA_eval/utt2systemID',
        'isstft': False,
}

la_cqt = {
        'train_scp': 'data/cqt/LA_train/feats_slicing.scp',
        'train_utt2index': 'data/cqt/LA_train/utt2index',
        'dev_scp': 'data/cqt/LA_dev/feats_slicing.scp',
        'dev_utt2index': 'data/cqt/LA_dev/utt2index',
        'dev_utt2systemID': 'data/cqt/LA_dev/utt2systemID',
        'scoring_dir': 'scoring/la_cm_scores/',
        'eval_scp': 'data/cqt/LA_eval/feats_slicing.scp',
        'eval_utt2index': 'data/cqt/LA_eval/utt2index',
        'eval_utt2systemID': 'data/cqt/LA_eval/utt2systemID',
        'isstft': False,
}

debug_feats = {
        'train_scp': 'data/debug_samples/feats_slicing.scp',
        'train_utt2index': 'data/debug_samples/utt2index',
        'dev_scp': 'data/debug_samples/feats_slicing.scp',
        'dev_utt2index': 'data/debug_samples/utt2index',
        'dev_utt2systemID': 'data/debug_samples/utt2systemID',
        'scoring_dir': 'scoring/la_cm_scores/',
        'eval_scp': 'data/debug_samples/feats_slicing.scp',
        'eval_utt2index': 'data/debug_samples/utt2index',
        'eval_utt2systemID': 'data/debug_samples/utt2systemID', 
        'isstft': False,
}

data_prepare = {
        'la_spec': la_spec,
        'la_cqt': la_cqt,
        'la_lfcc': la_lfcc,
        'debug_feats': debug_feats,
}

