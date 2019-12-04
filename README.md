# PoetryGenerator

This file should give the steps which will exactly reproduce the numbers, tables and figures in your report.

#### Data File: Wrangled Data: <filename1>, corpus_CGR.txt
  
#### Code File 1: Data Preprocessing: <filename1.py>, <filename2.py>

#### Code File 2: Single-Layer LSTM: <filename.py>

#### Code File 3: CharRNN: <filename.py>
#### Code File 4: Perplexity: Perplexity.ipynb
1. Use NLTK package to create n-gram models (n=1,2,3,4) based on the corpus (an n-gram list and n-gram frequency list); 

2. Function Ngram_mle: Calculate the MLE with smoothing on each n-gram; 

3. Function Ngram_perplexity: Use Function Ngram_mle to generate MLE with smoothing and use MLE result to calculate the perplexities on a piece of sentence;

4. Function total_Ngram_perpelxity: Based on perplexity on each sentence generated by Function Ngram_perplexity, calculate the total perplexity of the entire poem;
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/perplexity_formula.PNG "Perplexity Formulas")

5. Use total_Ngram_perplexity function, calculate the n-gram perplexities on each of the poem generated by two models and make comparison between the two models, create the perplexity table. 
![alt text](https://github.com/sayayangnu/PoetryGenerator/blob/master/perplexity_table.PNG "Perplexity Table")

