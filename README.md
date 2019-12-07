# PoetryGenerator

This file should give the steps which will exactly reproduce the numbers, tables and figures in your report.

#### Section 1: Data Preprocessing and Single-Layer LSTM
#### File: Normal_lstm.ipynb
This file preprocesses the data and builds a single-layer LSTM model to generate short poem sentences when giving the begin word and the intended length of sentence. It follows the following steps: 


1. Gettng Input. The input raw text is pre-processed as follows:

    a) Clean foreword and backwords

    b) Split into paragraphs

    c) Clean titles
    
    d) Output the cleaned poetry collection file: **corpus.txt**

2. Process Data.  For this model, we scraped Don Juan collection from http://www.gutenberg.org/cache/epub/21700/pg21700.txt and preprocessed it. Then we processed the data as follows:
    a) Poems are tokenizing with keras tokenizer by paragraph. Thus, a collection of poems turned into a list of tokens

    b) Each list of tokens is turned into a vector with the keras text_to_sequence function. Thus, a list of tokens turned into a list of vectors
  
    c) For each vector with n elements, use the 1 to n-1 element (feature) to predict the n element (target). Thus, a vector with n element will be converted to n-1 vectors of feature and target pairs. 
  
    d) Split the vectors got from step c) into features and target.
  
  
3. Build Model 
The model used is LSTM (Long short-term memory), which is one type of RNN (recurrent neural network). The model structure is explianed as follows:

    a) The vectors are first embedded into a dimention of 128 

    b) The embedded vectors are fed into a LSTM layer with 64 cells, with a Relu activation function

    c) The result from above is converted back to categories (representing texts) with a softmax layer 
  
The model uses Adam as an optimizer and an 'accuracy' metric keeps track of the model performance. 


4. Output

The model generates sentances when giving a) a begin word b) the length of the sentence. After training the model for 1437 epoch, the model's performance starts to converge. A sample sentence generated is as follows:

      Input: Night

      Length: 8 

      Output: " Night was a wicked world having been tost " 


5. Analysis: Accuracy / Loss vs. Epochs

Plot the accuracy vs. epochs plot and loss vs. epochs plot to observe the converging rate of the model.
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/img/accuracy1.jpeg "SingleLSTM Accuracy vs. Epochs")
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/img/loss1.jpeg "SingleLSTM Loss vs. Epochs")

#### Section 2: CharRNN:
#### File: 
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/img/CharRNN.jpeg "CharRNN Accuracy vs. Epochs")

x. Analysis: 
Plot the accuracy vs. epochs plot and loss vs. epochs plot to observe the converging rate of the model.
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/img/accuracy2.jpeg "CharRNN Accuracy vs. Epochs")
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/img/loss2.jpeg "CharRNN Loss vs. Epochs")

#### Section 3: Perplexity: 
#### File: Perplexity.ipynb
During this phase we evaluate the language models with perplexity. It follows the below steps:

1. Use NLTK package to create n-gram models (n=1,2,3,4) based on the corpus (an n-gram list and n-gram frequency list); 

2. Ngram_mle: Calculate the MLE on each n-gram; 

3. Ngram_perplexity: Calculate the perplexities on a piece of sentence;

4. total_Ngram_perpelxity: Based on Ngram_perplexity on each sentence, calculate the total perplexity of the entire poem;

5. Use total_Ngram_perplexity function, calculate the n-gram perplexities on each of the poem generated by two models and make comparison between the two models, create the perplexity table. 
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/img/perplexity_table.PNG "Perplexity Table")

