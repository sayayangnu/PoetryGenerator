#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from keras.models import model_from_json


# In[4]:


# This function slices the unwanted text (introduction, etc.) at the beginning of the txt file 
# It takes a string 'my_str', and delete everything before the specified 'sub' 
def slicer_front(my_str,sub):
  index=my_str.find(sub)
  if index !=-1 :
        return my_str[index:] 
  else :
        raise Exception('Sub string not found!')


# In[5]:


# This function slices the unwanted text (introduction, etc.) at the end of the txt file 
def slicer_back(my_str,sub):
  index=my_str.find(sub)
  if index !=-1 :
        return my_str[:index] 
  else :
        raise Exception('Sub string not found!')


# In[6]:


# write a function to check distribution of each 'poem' after the split 
def see_len_dist(corpus):
    get_len_dist = []
    for i in corpus:
        get_len_dist.append(len(i))
    ax = sns.distplot(get_len_dist)
        # further stuff
    return ax


# In[7]:


# write a function to check distribution of 'very short item' after the split 
def see_short_len_dist(corpus):
    get_len_dist = []
    for i in corpus:
        get_len_dist.append(len(i))
    ax = sns.distplot([i for i in get_len_dist if i<300])
        # further stuff
    return ax


# In[8]:


# writ a function to read in data from a url 
def get_soup(target_url):
    r = requests.get(target_url)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup


# ### Read Data: for now, only run on Dyllan Thomas dataset

# In[9]:


with open('DylanThomas.txt',"r") as f:
    DT_raw = f.read().split('\n\n\n')
len(DT_raw)


# In[10]:


DT_c = [i for i in DT_raw if len(i)>200]
len(DT_c)


# In[11]:


CP_raw = get_soup('https://raw.githubusercontent.com/tfavory/pmlg-poem-generator/master/model_training/corpus.txt')


# In[12]:


CP = CP_raw.get_text().split('\n\n\n\n')
len(CP)


# In[13]:


CP_c = [i for i in CP if len(i)>200]
len(CP_c)


# In[14]:


SE_raw = get_soup('http://www.gutenberg.org/files/1934/1934-0.txt')


# In[15]:


# Clean preface, conclusion and titles
SE_txt = slicer_back(slicer_front(SE_raw.get_text(),'How sweet is the shepherd'),'***END OF THE PROJECT GUTENBERG') # Delete the preface and conclusion


# In[16]:


# the title are all capitalized, get rid of capitalized words
# split the text into each poem
SE = []
for i in SE_txt.split('\r\n\r\n\r\n\r\n'):
    i_c = re.sub('[A-Z \d\W]+\r\n','', i)
    SE.append(i_c)


# In[17]:


SE_c = [i for i in SE if len(i)>100]
len(SE_c)


# In[18]:


DE_raw = get_soup('http://www.gutenberg.org/cache/epub/8789/pg8789.txt')


# In[19]:


# Clean preface, conclusion and titles
DE_txt = slicer_back(slicer_front(DE_raw.get_text(),'IN the midway of this our'),'End of Project Gutenberg') # Delete the preface and conclusion


# In[20]:


# the title are all capitalized, get rid of capitalized words
# split the text into each poem
DE = []
for i in DE_txt.split('\r\n\r\n\r\n'):
    i_c = re.sub('[A-Z \d\W]+\r\n','', i)
    DE.append(i_c)
len(DE)


# In[21]:


DE_c = [i for i in DE if len(i)>10]
len(DE_c)


# In[22]:


DJ = get_soup('http://www.gutenberg.org/cache/epub/21700/pg21700.txt')


# In[23]:


# Clean preface, conclusion and titles
DJ_txt = slicer_back(slicer_front(DJ.get_text(),'I want a hero: an uncommon want,'),'End of the Project Gutenberg EBook') # Delete the preface and conclusion


# In[24]:


DJ = DJ_txt.split('\r\n\r\n')


# In[25]:


# append sentances
DJ_c = [i for i in DJ if len(i)> 50]


# In[26]:


StaryB = get_soup('http://www.gutenberg.org/cache/epub/6524/pg6524.txt')


# In[27]:


# Clean preface, conclusion and titles
# The titles are numbers
StaryB_txt = slicer_back(slicer_front(StaryB.get_text(),'Stray birds of summer come to my window to sing and fly away.'),'End of the Project Gutenberg EBook') # Delete the preface and conclusion
StaryB_txt = re.sub('[1-9]\d*','\r\n\r\n\r\n\r\n',StaryB_txt) # Clean the titles


# In[28]:


StaryB = StaryB_txt.split('\r\n\r\n\r\n\r\n')


# In[29]:


# append sentances
StaryB_c = [i for i in StaryB if len(i)> 10]


# In[30]:


GH = get_soup('http://www.gutenberg.org/cache/epub/30488/pg30488.txt')


# In[32]:


# Clean preface and conclusion
GH_txt = slicer_back(slicer_front(GH.get_text(),'I swayed upon the gaudy stern'),'One cannot begin it too soon.') # Delete the preface and conclusion
# the titles are all capitalized, get rid of capitalized words
GH_txt = re.sub('[A-Z \d\W]+\r\n\r\n','\r\n\r\n\r\n\r\n\r\n',GH_txt) # Clean the titles


# In[33]:


# split the text into each poem
GH = GH_txt.split('\r\n\r\n\r\n\r\n\r\n')


# In[34]:


GH_c = [i for i in GH if len(i)>150]
len(GH_c)


# In[35]:


JRL = get_soup('http://www.gutenberg.org/files/38520/38520-0.txt')


# In[36]:


# Clean preface, conclusion and titles
JRL_txt = slicer_back(slicer_front(JRL.get_text(),'If some small savor creep into my rhyme'),'But is lord of the earldom as much as he.') # Delete the preface and conclusion
# the titles are all capitalized, get rid of capitalized words
JRL_txt = re.sub('[A-Z \d\W]+\r\n\r\n','\r\n\r\n\r\n\r\n\r\n',JRL_txt) # Clean the titles
JRL_txt = re.sub('[1-9]\d*\.','',JRL_txt) # Clean the numbers


# In[37]:


# split the text into each poem
JRL = JRL_txt.split('\r\n\r\n\r\n\r\n\r\n')


# In[38]:


JRL_c = [i for i in JRL if len(i)>150]
len(JRL_c)


# In[39]:


CGR = get_soup('http://www.gutenberg.org/cache/epub/19188/pg19188.txt')


# In[40]:


# Clean preface, conclusion and titles
CGR_txt = slicer_back(slicer_front(CGR.get_text(),'Morning and evening'),'We trust to Thee.') # Delete the preface and conclusion

# Some of the titles are capitalized, some of the titles are numbers. 
# Get rid of capitalized words and numbers
CGR_txt = re.sub('[A-Z \d\W]+\r\n\r\n','\r\n\r\n\r\n\r\n\r\n',CGR_txt) # Clean the titles
CGR_txt = re.sub('[1-9]\d*\.','\r\n\r\n\r\n\r\n\r\n',CGR_txt) # Clean the numbers


# In[41]:


# split the text into each poem
CGR = CGR_txt.split('\r\n\r\n\r\n\r\n\r\n')


# In[42]:


CGR_c = [i for i in CGR if len(i)>100]


# In[43]:


corpus = DT_c+ CP_c+ SE_c+ DE_c+ DJ_c + StaryB_c + GH_c + JRL_c + CGR_c


# In[44]:


len(corpus)
### This is a TEST
corpus = corpus[:500]


# ### Preprocessing

# In[45]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len( tokenizer.word_index ) + 1
total_words


# In[46]:


input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)


# In[47]:


sequence_lengths = list()
for x in input_sequences:
    sequence_lengths.append( len( x ) )
max_sequence_len = max( sequence_lengths )
max_sequence_len


# In[48]:


input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_sequence_len+1, padding='pre'))
x, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)


# 
# dropout_rate = 0.2
# activation_func = keras.activations.relu
# 
# SCHEMA = [
# 
#     Embedding( total_words , 64, input_length=max_sequence_len ),
#     LSTM( 64 ) ,
#     Dropout(dropout_rate),
#     Dense( 32 , activation=activation_func ) ,
#     Dropout(dropout_rate),
#     Dense( total_words, activation=tf.nn.softmax )
# 
# ]
# model = keras.Sequential(SCHEMA)
# model.compile(
#     optimizer=keras.optimizers.Adam() ,
#     loss=keras.losses.categorical_crossentropy ,
#     metrics=[ 'accuracy' ]
# )
# model.summary()

# In[59]:


# clm new
dropout_rate = 0.2
activation_func = keras.activations.relu

SCHEMA = [

    Embedding( total_words , 512, input_length=max_sequence_len ),
    LSTM( 128, return_sequences = True ) ,
    Dropout(dropout_rate),
    LSTM( 128, return_sequences = True ),
    Dropout(dropout_rate),
    LSTM( 64 ),
    Dropout(dropout_rate),
    Dense( 32 , activation=activation_func ) ,
    Dropout(dropout_rate),
    Dense( total_words, activation=tf.nn.softmax )

]
model = keras.Sequential(SCHEMA)
model.compile(
    optimizer=keras.optimizers.Adam() ,
    loss=keras.losses.categorical_crossentropy ,
    metrics=[ 'accuracy' ]
)
model.summary()


# In[81]:


from keras.callbacks import ModelCheckpoint
checkpoint = [ModelCheckpoint(filepath='models.hdf5',period=20)]


# In[ ]:


model.fit(
    x,
    y,
    batch_size=32 ,
    epochs=200,
    callbacks=checkpoint
)


# from keras.models import load_model

# m1 = load_model('models.hdf5')

# In[73]:


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# def predict(seed_text , seed=10 ):
# 
#     for i in range( seed ):
# 
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=
#         max_sequence_len , padding='pre')
#         predicted = model.predict_classes(token_list, verbose=0 )
#         output_word = ""
#         for word, index in tokenizer.word_index.items():
#             if index == predicted:
#                 output_word = word
#                 break
#         seed_text += " " + output_word
# 
#     return seed_text

# print( 
#   predict( 
#     input( 'Enter some starter text ( I want ... ) : ') , 
#     int( input( 'Enter the desired length of the generated sentence : '))  
#   ) 
# )

# from keras.models import model_from_json
# ##### load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model =  keras.models.model_from_json(loaded_model_json)
# ##### load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# def predict(seed_text , seed=10 ):
# 
#     for i in range( seed ):
# 
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=
#         max_sequence_len , padding='pre')
#         predicted = loaded_model.predict_classes(token_list, verbose=0 )
#         output_word = ""
#         for word, index in tokenizer.word_index.items():
#             if index == predicted:
#                 output_word = word
#                 break
#         seed_text += " " + output_word
# 
#     return seed_text
# 
# print( 
#   predict( 
#     input( 'Enter some starter text ( I want ... ) : ') , 
#     int( input( 'Enter the desired length of the generated sentence : '))  
#   ) 
# )
