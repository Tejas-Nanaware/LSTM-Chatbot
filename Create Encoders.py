#!/usr/bin/env python
# coding: utf-8

# In[1]:


import codecs
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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


# In[2]:


with codecs.open("./inputs/encoder_inputs.txt", "rb", encoding="utf-8", errors="ignore") as f:
    lines = f.read().split("\n")
    encoder_text = []
    for line in lines:
        data = line.split("\n")[0]
        encoder_text.append(data)


# In[3]:


encoder_text


# In[4]:


with codecs.open("./inputs/decoder_inputs.txt", "rb", encoding="utf-8", errors="ignore") as f:
    lines = f.read().split("\n")
    decoder_text = []
    for line in lines:
        data = line.split("\n")[0]
        decoder_text.append(data)


# In[ ]:


decoder_text


# In[ ]:


# Check dictionary size
full_text = encoder_text + decoder_text


# In[ ]:


dictionary = []
for text in full_text:
    words = text.split()
    for i in range(0, len(words)):
        if words[i] not in dictionary:
            dictionary.append(words[i])


# In[ ]:


len(dictionary)


# In[ ]:


VOCAB_SIZE = 29999
tokenizer = Tokenizer(num_words=VOCAB_SIZE)


# In[ ]:


tokenizer.fit_on_texts(full_text)
word_index = tokenizer.word_index
len(word_index)


# In[ ]:


index2word = {}
for k, v in word_index.items():
    if v < (VOCAB_SIZE+1):
        index2word[v] = k
    if v > (VOCAB_SIZE+1):
        continue


# In[ ]:


index2word


# In[ ]:


word2index = {}
for k, v in index2word.items():
    word2index[v] = k


# In[ ]:


word2index


# In[ ]:


len(word2index) == len(index2word)


# In[ ]:


len(word2index)


# In[ ]:


encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
decoder_sequences = tokenizer.texts_to_sequences(decoder_text)


# In[ ]:


encoder_sequences


# In[ ]:


decoder_sequences


# In[ ]:


for seqs in decoder_sequences:
    for seq in seqs:
        if seq > 29999:
            print(seq)
            break


# In[ ]:


VOCAB_SIZE = len(index2word) + 1
VOCAB_SIZE


# In[ ]:


MAX_LEN = 20
num_samples = len(encoder_sequences)
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")


# In[ ]:


encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')


# In[ ]:


for i, seqs in enumerate(decoder_input_data):
    for j, seq in enumerate(seqs):
        if j > 0:
            decoder_output_data[i][j][seq] = 1.


# In[ ]:


decoder_output_data.shape


# In[ ]:


word_embeddings = {}
line_num = 0
with open('./glove_vectors/glove.840B.300d.txt', encoding='utf-8') as glove_file:
    for line in glove_file:
        val = line.split()
        line_num += 1
        try:
            numpy_array = np.asarray(val[1:], dtype='float32')
            word_embeddings[val[0]] = numpy_array
        except:
            print('Ignoring line', line_num)


# In[ ]:


len(word_embeddings)


# In[ ]:


word_vec_dimension = 300
def create_word_to_vec(word_vec_dimension, word_index):
    vec_matrix = np.zeros((len(word_index)+1, word_vec_dimension))
    for word, i in word_index.items():
        vec_embedding = word_embeddings.get(word)
        if vec_embedding is not None:
            vec_matrix[i] = vec_embedding
    return vec_matrix


# In[ ]:


vec_matrix = create_word_to_vec(word_vec_dimension, word2index)


# In[ ]:


keras_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=word_vec_dimension, 
                             trainable=True)
keras_embedding.build((None,))
keras_embedding.set_weights([vec_matrix])


# In[ ]:


# Applying the LSTM
HIDDEN_DIM=1000


# In[ ]:


enc_input = Input(shape=(MAX_LEN, ), dtype='int32')
enc_vec = keras_embedding(enc_input)
enc_LSTM = LSTM(HIDDEN_DIM, return_state=True)
enc_output, enc_hidden, enc_cell_state = enc_LSTM(enc_vec)


# In[ ]:


dec_input = Input(shape=(MAX_LEN, ), dtype='int32')
dec_vec = keras_embedding(dec_input)
dec_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
dec_output, dec_hidden, dec_cell_state = dec_LSTM(dec_vec, initial_state=[enc_hidden, enc_cell_state])


# In[ ]:


lstm_output = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(dec_output)
model = Model([enc_input, dec_input], lstm_output)
model.compile(optimizer=Adam(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])


# In[ ]:


my_epochs = 5
lstm_fit = model.fit([encoder_input_data, decoder_input_data], 
                     decoder_output_data, epochs=my_epochs, 
                     batch_size = 32)
model.save('./models/lstm.h5')


# In[ ]:


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




