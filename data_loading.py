import os
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeMask
WANTED_SR = 16000

_audio_augmentations = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeMask(min_band_part=0.02, max_band_part=0.1, p=0.5)
])

@tf.py_function(Tout=tf.float32)
def _audio_augmentation(audio):
    augmented = _audio_augmentations(audio.numpy(), WANTED_SR)
    return augmented

# audio loading and preprocessing
def _load_audio(file_path: str, wanted_sr : int = WANTED_SR) -> tf.Tensor:
    """
    Load audio file from file path and return it as a tensor.
    The audio file will be loaded as a mono-channel audio with a sample rate of 16kHz.
    """
    audio = tf.io.read_file(file_path)
    wav, sr = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav = tfio.audio.resample(wav, tf.cast(sr, tf.int64), wanted_sr)
    return wav


def _load_dataset(pos_path: str, neg_path: str) -> tf.data.Dataset:
    # Load the audio files
    pos_ds = tf.data.Dataset.list_files(os.path.join(pos_path, '*.wav'))
    neg_ds = tf.data.Dataset.list_files(os.path.join(neg_path, '*.wav'))

    # Load the audio files as tensors and add labels
    pos_samples = tf.data.Dataset.zip((pos_ds, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos_ds)))))
    neg_samples = tf.data.Dataset.zip((neg_ds, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg_ds)))))

    # Combine the datasets
    dataset = pos_samples.concatenate(neg_samples)

    return dataset

def _preprocess_sample_audio(file_path: str, label: int) -> tuple[tf.Tensor, int]:
    """
    Load the audio sample, make it exactly 3 seconds long and add the label
    """
    audio = _load_audio(file_path)
    # Make the audio exactly 3 seconds long
    audio = tf.cond(tf.shape(audio)[0] < 48000, lambda: tf.pad(audio, [[0, 48000 - tf.shape(audio)[0]]]), lambda: audio[:48000])
    return audio, label

def _preprocess_sample_to_spectogram(audio: tf.Tensor, label: int) -> tuple[tf.Tensor, int]:
    """
    Preprocess the audio sample to a spectogram and add the label
    """
    # Compute the spectrogram
    spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.transpose(spectrogram, perm=[1, 0])
    spectrogram = tf.expand_dims(spectrogram, axis=2)

    # Add the label
    return spectrogram, label


def _preprocess_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    apply the augmentation pipeline to the dataset, then preprocess the samples
    """
    dataset = dataset.map(_preprocess_sample_audio)
    dataset = dataset.repeat(2)
    dataset = dataset.map(lambda x, y: (_audio_augmentation(x), y))
    dataset = dataset.map(_preprocess_sample_to_spectogram)
    return dataset


def get_augmented_data(positive_files_path: str, negative_files_path: str, train_test_ratio: float=0.8) -> tf.data.Dataset:
    """
    Load the data from the given file paths and return it as a tf.data.Dataset.
    The dataset will be shuffled and batched.
    """
    dataset = _load_dataset(positive_files_path, negative_files_path)
    dataset = _preprocess_dataset(dataset)

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(200).batch(32).prefetch(16)

    # Split the dataset into a training and a testing set
    train_size = int(len(list(dataset)) * train_test_ratio)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    return train_dataset, test_dataset
