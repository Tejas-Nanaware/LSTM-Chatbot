# Introduction
Create a LSTM for Seq2Seq dataset.

# Dataset
The dataset used is the Cornell Movie Dataset which can be downloaded from [here](http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip).  
For the vector representation, I have used glove vector which consists of 840B Tokens and 2.2M vocab, which are cased to obtain 300d vectors. The file for the vectors can be downloaded from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip).  
  
# Directory Structure
data
|--cornell movie-dialogs corpus
|--|--movie_lines.txt
glove_vectors
|--glove.840B.300d.txt
Data Preprocessing.ipynb
Create Encoders.ipynb
LSTM Model.ipynb

If you are executing the code, execute the notebooks in the above order.