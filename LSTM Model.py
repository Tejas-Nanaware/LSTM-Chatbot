#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

import matplotlib.pyplot as plt
# MacOS matplotlib kernel issue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[25]:


# Constants from Create Encoders
MAX_LEN = 20
VOCAB_SIZE = 30000
HIDDEN_DIM=1000
word_vec_dimension = 300


# In[26]:


# Load pickled variables that are required
vec_matrix = np.load('./pickle/vec_matrix.npy')
encoder_input_data = np.load('./pickle/encoder_input_data.npy')
decoder_input_data = np.load('./pickle/decoder_input_data.npy')


# In[27]:


print(vec_matrix.shape)
print(encoder_input_data.shape)
print(decoder_input_data.shape)


# In[28]:


# Create Decoder output data
num_samples = encoder_input_data.shape[0]
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")
for i, seqs in enumerate(decoder_input_data):
    for j, seq in enumerate(seqs):
        if j > 0:
            decoder_output_data[i][j][seq] = 1.
print(decoder_output_data.shape)


# In[29]:


keras_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=word_vec_dimension, 
                             trainable=True)
keras_embedding.build((None,))
keras_embedding.set_weights([vec_matrix])


# In[30]:


enc_input = Input(shape=(MAX_LEN, ), dtype='int32')
enc_vec = keras_embedding(enc_input)
enc_LSTM = LSTM(HIDDEN_DIM, return_state=True)
enc_output, enc_hidden, enc_cell_state = enc_LSTM(enc_vec)


# In[31]:


dec_input = Input(shape=(MAX_LEN, ), dtype='int32')
dec_vec = keras_embedding(dec_input)
dec_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
dec_output, dec_hidden, dec_cell_state = dec_LSTM(dec_vec, initial_state=[enc_hidden, enc_cell_state])


# In[32]:


lstm_output = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(dec_output)
model = Model([enc_input, dec_input], lstm_output)
model.compile(optimizer=Adam(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])


# In[33]:


my_epochs = 5
lstm_fit = model.fit([encoder_input_data, decoder_input_data], 
                     decoder_output_data, epochs=my_epochs, 
                     batch_size = 32)
model.save('./models/lstm.h5')


# In[26]:


train_acc = lstm_fit.history['acc']
train_loss = lstm_fit.history['loss']

# Plot the accuracies and losses
plt.figure(figsize=(16,6))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.title("Training Accuracy over epochs")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.title("Training Loss over epochs")
plt.grid()
plt.show()


# In[ ]:




