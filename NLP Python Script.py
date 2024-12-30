# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 09:35:59 2024

@author: macph
"""

# import the necessary libraries

# librosa module helps manage audio files
import librosa

# Import pytorch for deploying neural network models and natural language processing
import torch

# import wav2vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# load the raw audio file
audio, rate = librosa.load("PATH/OSR_us_000_0010_8k.wav", sr = 16000)

# audio is an array of the amplitudes of the waveform sampled at the rate
print(audio)

# print rate
print(rate)
#%%
# Importing Wav2Vec pretrained model and tokenizer from Meta
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Tokenize the raw audio data into a usable input for the model. The model takes in tensor data format
input_values = tokenizer(audio, return_tensors = "pt").input_values

# Storing logits gives a relative probability of what the token at each frame is
logits = model(input_values).logits

# picks the token with the highest score at each timestep
prediction = torch.argmax(logits, dim = -1)

# Passing the prediction to the tokenizer decode to get the transcription
transcription = tokenizer.batch_decode(prediction)[0]

#%%
# Printing the transcription
print(transcription)
#%%
