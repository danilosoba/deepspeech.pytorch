import os
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import scipy.signal
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError

class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        self.paths = path is not None and librosa.util.find_files(path)
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_src = load_audio(noise_path)
        noise_offset_fraction = np.random.rand()
        noise_dst = np.zeros_like(data)
        src_offset = int(len(noise_src) * noise_offset_fraction)
        src_left = len(noise_src) - src_offset
        dst_offset = 0
        dst_left = len(data)
        while dst_left > 0:
            copy_size = min(dst_left, src_left)
            np.copyto(noise_dst[dst_offset:dst_offset + copy_size],
                      noise_src[src_offset:src_offset + copy_size])
            if src_left > dst_left:
                dst_left = 0
            else:
                dst_left -= copy_size
                dst_offset += copy_size
                src_left = len(noise_src)
                src_offset = 0
        data += noise_level * noise_dst
        return data

class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_audio_mfcc(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        #win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        ## STFT
        #D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        #                 win_length=win_length, window=self.window)
        #spect, phase = librosa.magphase(D)
        ## S = log(S+1)
        #spect = np.log1p(spect)
        #spect = torch.FloatTensor(spect)
        #if self.normalize:
        #    mean = spect.mean()
        #    std = spect.std()
        #    spect.add_(-mean)
        #    spect.div_(std)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=40, n_fft=n_fft, hop_length=hop_length, n_mels=64)
        mfcc = torch.FloatTensor(mfcc)
        #print("MFCC OF TENTH FRAME:",mfcc[:, 9])
        if self.normalize:
            mean = mfcc.mean()
            std = mfcc.std()
            mfcc.add_(-mean)
            mfcc.div_(std)

        return mfcc

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath,
                 #labels,
                 normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        if len(ids[0]) == 3:
            self.speaker_labels = []
            for x in ids:
                self.speaker_labels.append(int(x[2]))
                del x[2]
            print("SPEAKER LABELS: " + str(len(self.speaker_labels)))
        else:
            self.speaker_labels = None
            print("SPEAKER LABELS: NONE")
        self.ids = ids
        self.size = len(ids)
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        speaker_label = self.speaker_labels[index]
        mfcc = self.parse_audio_mfcc(audio_path)
        #print("SPECT SIZE:",spect.size())
        #print("MFCC SIZE:",mfcc.size())
        return spect, speaker_label, mfcc

    def __len__(self):
        return self.size

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    mfccs = torch.zeros(minibatch_size, 1, 40, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    speaker_labels_returned = torch.IntTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        speaker_labels_returned[x] = sample[1]
        mfcc = sample[2]
        seq_length = tensor.size(1)
        seq_length_mfcc = mfcc.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        mfccs[x][0].narrow(1, 0, seq_length_mfcc).copy_(mfcc)
        input_percentages[x] = seq_length / float(max_seqlength)
    return inputs, input_percentages, speaker_labels_returned, mfccs

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                            augmented_filename,
                                                                            " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y

def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
