import os
import sys
import numpy as np
import librosa
import tensorflow as tf
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# if you need any imports you can do that here.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sample_rate = 48000
def load_wav(x, sample_rate=48000):
    '''This return the array values of audio with sampling rate of 48000 and Duration'''
    samples_, sample_rate = librosa.load(x, sr=sample_rate)
    non_silent = librosa.effects.split(samples_, frame_length=1024, hop_length=50)
    samples = np.concatenate([samples_[i:j] for i,j in non_silent])
    return samples

def load_wav_2(samples_, sample_rate=48000):
    '''This return the array values of audio with sampling rate of 48000 and Duration'''
    non_silent = librosa.effects.split(samples_, frame_length=1024, hop_length=50)
    samples = np.concatenate([samples_[i:j] for i,j in non_silent])
    return samples

def pad_sample(x, max_length=220000):
  if len(x)<max_length:
    return np.hstack((x, np.zeros(max_length-len(x))))
  else:
    return x[:max_length]

def convert_to_mfccs(x, sr=48000):
  mfccs = librosa.feature.mfcc(x, n_mfcc=39, sr=sr)
  # mfccs = librosa.power_to_db(S=mfccs, ref=np.max)
  return mfccs

def convert_to_melspectrogram(x, sr=48000):
  mfccs = librosa.feature.melspectrogram(x, n_mels=64, sr=sr)
  mfccs = librosa.power_to_db(S=mfccs, ref=np.max)
  return mfccs

def convert_to_delta(mfcc):
    '''converting to velocity'''
    delta = librosa.feature.delta(mfcc)
    # delta = librosa.power_to_db(S=delta, ref=np.max)
    return delta

def convert_to_delta_2(mfcc):
    '''converting to velocity'''
    delta_2 = librosa.feature.delta(mfcc, order=2)
    # delta_2 = librosa.power_to_db(S=delta_2, ref=np.max)
    return delta_2

# Zero Crossing Rate
def ZCR(data):
  zcr = librosa.feature.zero_crossing_rate(data)
  return zcr[0]

# RMS energy
def RMSE(data):
  rmse = librosa.feature.rms(data)
  return rmse[0]

def standardize(data, axis=0):
  data -= np.mean(data, axis=axis)
  data /= np.std(data, axis=axis)
  return data

def expand_dims(X_train_mfcc):
  a,b,c = X_train_mfcc.shape
  res = np.zeros((a,b,c,3))
  for i in range(3):
    res[:,:,:,i] = X_train_mfcc
  return res

def process_data(file):
  file_path = file
  x_aug = load_wav(file_path)
  x_aug_pad = pad_sample(x_aug)

  x_zcr = np.array([ZCR(x_aug_pad)[np.newaxis, ...]])
  x_rmse = np.array([RMSE(x_aug_pad)[np.newaxis, ...]])
  x_mfcc = np.array([convert_to_mfccs(x_aug_pad)])
  x_delta = convert_to_delta(x_mfcc)
  x_delta_2 = convert_to_delta_2(x_mfcc)
  x_melspectrogram = np.array([convert_to_melspectrogram(x_aug_pad)])

  x_img = np.concatenate((x_melspectrogram, x_mfcc, x_delta,
                          x_delta_2, x_zcr, x_rmse), axis=1)

  x_img = expand_dims(x_img)
  return x_img

def load_model():
  model = tf.keras.models.load_model('./resnet50_covid.h5')
  return model

# file_name = sys.argv[1]
# if '.wav' not in file_name:
#   print('please pass an audio file name ending with .wav')
#
# else:
#   file_name = './'+file_name
#   print('file loaded... processing now...')
#   x_img = process_data(file_name)
#   print('file processed... loading model...')
#   model = load_model()
#   print('model loaded... predicting...')
#   prob = model.predict(x_img)
#   print('*'*30)
#   print('probability of covid-19:', prob)
#   print('*'*30)
