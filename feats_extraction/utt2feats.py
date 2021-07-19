import numpy as np
import librosa
from librosa.feature import delta
import json

from feats_extraction.generic import stft, cqt, load_wav, preemphasis, load_wav_snf
from feats_extraction.LFCC import lfcc

def logpowcqt(wav_path, config_json, ref=1.0, amin=1e-30, top_db=None):

    wav = load_wav_snf(wav_path)
    with open(config_json) as json_file:
        config = json.load(json_file)
    
    # print(config)
    if config['pre_emphasis'] is not None:
        wav = preemphasis(wav, k=config['pre_emphasis'])
    cqtfeats = cqt(wav, sr=config['sample_rate'], hop_length=config['hop_length'], n_bins=config['n_bins'], bins_per_octave=config['bins_per_octave'], window=config['window'], fmin=config['fmin'])
    magcqt = np.abs(cqtfeats)
    powcqt = np.square(magcqt)
    logpowcqt = librosa.power_to_db(powcqt, ref, amin, top_db)
    return logpowcqt

def logmagspec(wav_path, sr=16000, n_fft=512, hop_length=160, win_length=400, window="hann", pre_emphasis=None):

    wav = load_wav_snf(wav_path)
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    mag_spec = np.abs(spec)
    mag_spec[mag_spec <= 1e-30] = 1e-30
    lms = 10 * np.log10(mag_spec)
    return lms


def logpowspec(wav_path, sr=16000, n_fft=512, hop_length=160, win_length=400, window="hann", pre_emphasis=0.97, ref=1.0, amin=1e-30, top_db=None):
    """Compute log power magnitude spectra (logspec).

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Returns:
        D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.
    amin : float > 0 [scalar], ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
        minimum threshold for `abs(S)` and `ref`
    
    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``
    """
    wav = load_wav_snf(wav_path)
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    # spec = spec[:-2, :]  # TODO: check why there are two abnormal frames.
    mag_spec = np.abs(spec)
    powspec = np.square(mag_spec)
    logpowspec = librosa.power_to_db(powspec, ref, amin, top_db)
    return logpowspec

def logpowspec_multichannel(wav_path, channel, sr=16000, n_fft=512, hop_length=160, win_length=400, window="hann", pre_emphasis=0.97, ref=1.0, amin=1e-30, top_db=None):

    wav = load_wav_snf(wav_path)
    try:
        wav = wav[:, channel]
    except:
        wav = wav
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    # spec = spec[:-2, :]  # TODO: check why there are two abnormal frames.
    mag_spec = np.abs(spec)
    powspec = np.square(mag_spec)
    logpowspec = librosa.power_to_db(powspec, ref, amin, top_db)
    return logpowspec


def extract_lfcc(wav_path, config_json):

    wav = load_wav_snf(wav_path)
    with open(config_json) as json_file:
        config = json.load(json_file)

    # print(config)
    lfccfeats = lfcc(wav, fs=config['fs'], num_ceps=config['num_ceps'], win_len=config['win_len'], win_hop=config['win_hop'], nfilts=config['nfilts'], nfft=config['nfft'])
    lfccdelta = delta(lfccfeats, order=1)
    lfccdelta2 = delta(lfccfeats, order=2)

    feats = np.concatenate((lfccfeats, lfccdelta, lfccdelta2), axis=-1)
    return feats

def extract_stft(wav_path, config_json):
    
    wav = load_wav_snf(wav_path)
    with open(config_json) as json_file:
        config = json.load(json_file)

    stftfeats = stft(wav, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=config['win_length'], window='hann')
    real = np.real(stftfeats)
    imag = np.imag(stftfeats)
    feats_cat = np.concatenate((real, imag), axis=-1)

    return feats_cat

if __name__ == '__main__':
    wav_path = '/apdcephfs/private_nenali/lixu/Data_Source/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0000001.flac'
    print(wav_path)
    lpcqt = logpowcqt(wav_path)
    print(lpcqt.shape)

