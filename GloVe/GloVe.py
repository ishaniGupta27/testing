
# coding: utf-8

# In[39]:


import numpy as np 
import json
from random import shuffle
import zipfile
import sys
import scipy as sp
from scipy import sparse
import logging
from math import log


# In[18]:


with open('configGloVe.json') as json_data_file:
    ConfigJsondata = json.load(json_data_file)


# In[19]:


#builing vocab --> setting vocab size and making hot vectors !
with open("vocab.txt","r") as f:
    data = f.readlines()
vocab_file  = open("vocab.txt", "r") #just wanna read the file
word2int={}#hash map
int2word={}#hash map

vocabWordsDict=[] #Array 

for word in data:
        vocabWordsDict.append(word.split('\n')[0])


vocabSize=len(vocabWordsDict)#How many words we have in our vocabl

for i,word in enumerate(vocabWordsDict):
   
   word2int[word] = i
   int2word[i] = word 
print "The Vocab Size is :::"
print vocabSize
#print word2int


# In[36]:


def read_corpus_data(filename):
    """
    input parameters: filename from which th ecorpus need ot be read
    Reading the corpus from the zipfile
    """

    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data
corpus_raw = read_corpus_data('text8.zip')
#corpus= "abandon abandonment abashed abandonment abbreviate"
#corpus_raw=corpus_raw.split()
corpus_size=len(corpus_raw)
sys.stdout.write("\n Our Corpus size is-------------\n")
sys.stdout.write(str(len(corpus_raw))) #len(data)
sys.stdout.write("\n--------------------------------\n")


# In[74]:


def build_cooccur(vocabWordsDict, corpus, WINDOW_SIZE):
    """
    input parameters: filename from which th ecorpus need ot be read
    Reading the corpus from the zipfile
    """
    cooccurrences = sparse.lil_matrix((vocabSize, vocabSize),dtype=np.float64)
    for word_index, word in enumerate(corpus_raw):
        if(word_index>=WINDOW_SIZE & word_index<(corpus_size)):
            counter=0;
            while counter<WINDOW_SIZE:
                if word in vocabWordsDict:
                    distFromCenterWord=counter+1
                    incrementVal = 1.0 / float(distFromCenterWord)
                    if corpus_raw[word_index-(counter+1)] in vocabWordsDict:
                        nb_word=corpus_raw[word_index-(counter+1)]
                        cooccurrences[word2int[word],word2int[nb_word]]+=incrementVal;
                    if (word_index+counter+2)<(corpus_size-2):
                        if corpus_raw[word_index+counter+2] in vocabWordsDict:
                            nb_word2=corpus_raw[word_index+counter+2]
                            cooccurrences[word2int[word],word2int[nb_word2]]+=incrementVal;
                counter=counter+1
    return cooccurrences


# In[75]:


cooccurrences= build_cooccur(vocabWordsDict[0:5],corpus_raw,2)


# In[79]:


def fetchCost(vocab, mainVec, contextVec, mainBias, contextBias, gradsq_W_main, gradsq_W_context,gradsq_mainBias, gradsq_contextBias, cooccurrence, learning_rate, x_max, learnRate):
    wght=1
    if cooccurrence < x_max:
        wght = (cooccurrence / x_max) ** learnRate 
    else:
        wght= 1

    cost_inner = (mainVec.dot(contextVec)+ mainBias + contextBias- log(cooccurrence))

    cost = ((cost_inner ** 2)* wght) 

    
    grad_main = wght * cost_inner * contextVec
    grad_context = wght * cost_inner * mainVec

    # Computing the gradients for bias terms
    grad_bias_main = wght * cost_inner
    grad_bias_context = wght * cost_inner

    # Doing Stochastic Gradient Descent
    mainVec =mainVec- (learning_rate * grad_main / np.sqrt(gradsq_W_main))
    contextVec =contextVec- (learning_rate * grad_context / np.sqrt(gradsq_W_context))

    mainBias =mainBias- (learning_rate * grad_bias_main / np.sqrt(gradsq_mainBias))
    contextBias =contextBias- (learning_rate * grad_bias_context / np.sqrt(
            gradsq_contextBias))

    # Update squared gradient sums
    gradsq_W_main += np.square(grad_main)
    gradsq_W_context += np.square(grad_context)
    gradsq_mainBias += grad_bias_main ** 2
    gradsq_contextBias += grad_bias_context ** 2

    parameters = {"cost": cost,
                  "mainVec": mainVec,
                  "contextVec": contextVec,
                  "mainBias": mainBias,
                  "contextBias": contextBias,
                  "gradsq_W_main": gradsq_W_main,
                  "gradsq_W_context": gradsq_W_context,
                  "gradsq_mainBias": gradsq_mainBias,
                  "gradsq_contextBias": gradsq_contextBias,}
    return parameters
    


# In[80]:
def initialize_parameters(n_x, n_h):
    """
    Argument:
    n_x -- Size of Vocab
    n_h -- Dimension Size
    Returns:
    params -- python dictionary containing your parameters:
                    W_i -- the word vector for the main word i in the co-occurrence matrix
                    W_j -- the word vector for the context word in the co-occurrence matrix
                    bias_i -- bias vector for main word 
                    bias_j -- bias vector for context word
                    W_i_grad_sq -- a matrix storing the squared gradient history for the main word vector (for use in the AdaGrad update)
                    W_j_grad_sq -- a matrix gradient history for the context word vector
                    bias_i_grad_sq -- a vector for the main word bias
                    bias_j_grad_sq --  a vector history for the context word bias
    """
    W_i = np.random.uniform(low=-0.5, high=0.5, size=(n_x, n_h)) / float(n_x + 1)
    W_j = np.random.uniform(low=-0.5, high=0.5, size=(n_x, n_h)) / float(n_x + 1)
    
    bias_i = (np.random.rand(n_x) - 0.5) / float(n_x + 1) 
    bias_j = (np.random.rand(n_x) - 0.5) / float(n_x + 1) 
   

    W_i_grad_sq = np.ones((n_x , n_h),dtype=np.float64)
    W_j_grad_sq = np.ones((n_x , n_h),dtype=np.float64)
   
    bias_i_grad_sq = np.ones(n_x, dtype=np.float64)
    bias_j_grad_sq = np.ones(n_x, dtype=np.float64)
    
    assert (W_i.shape == (n_x, n_h)) 
   
    
    parameters = {"W_i": W_i,
                  "W_j": W_j,
                  "bias_i": bias_i,
                  "bias_j": bias_j,
                  "W_i_grad_sq": W_i_grad_sq,
                  "W_j_grad_sq": W_j_grad_sq,
                  "bias_i_grad_sq": bias_i_grad_sq,
                  "bias_j_grad_sq": bias_j_grad_sq}    
    return parameters

def modelGloVe(vocab, cooccurrences,vector_size,iterations, learning_rate):
    
    parameters=initialize_parameters(vocabSize,vector_size)

    W_i = parameters["W_i"]
    W_j = parameters["W_j"] 
    bias_i = parameters["bias_i"]
    bias_j = parameters["bias_j"] 
    W_i_grad_sq = parameters["W_i_grad_sq"]
    W_j_grad_sq = parameters["W_j_grad_sq"]
    bias_i_grad_sq = parameters["bias_i_grad_sq"]
    bias_j_grad_sq = parameters["bias_j_grad_sq"]
  
    cx = sp.sparse.coo_matrix(cooccurrences)
    #print data.shape
    
    for i in range(iterations):
    	global_cost=0
        print("\tBeginning iteration %i..", i)
    	for i_main, i_context, cooccurrence in zip(cx.row, cx.col, cx.data):
	        
	        parameters = fetchCost(vocab,W_i[i_main], W_j[i_context],bias_i[i_main],bias_j[i_context],W_i_grad_sq[i_main], W_j_grad_sq[i_context],bias_i_grad_sq[i_main], bias_j_grad_sq[i_context],cooccurrence,learning_rate,ConfigJsondata["hyperparameters"]["MaxX"],ConfigJsondata["hyperparameters"]["alpha"])
	        
	        global_cost += 0.5 * parameters["cost"]
	        print 'Cost'
                print global_cost
                W_i[i_main]=parameters["mainVec"]
	        W_j[i_context]=parameters["contextVec"]
	        bias_i[i_main]=parameters["mainBias"]
	        bias_j[i_context]=parameters["contextBias"]
	        W_i_grad_sq[i_main]=parameters["gradsq_W_main"]
	        W_j_grad_sq[i_context]=parameters["gradsq_W_context"]
	        bias_i_grad_sq[i_main]=parameters["gradsq_mainBias"]
	        bias_j_grad_sq[i_context]=parameters["gradsq_contextBias"]

    return W_i
        

# In[ ]:


W = modelGloVe(vocabWordsDict, cooccurrences,(ConfigJsondata["hyperparameters"]["vecSize"]),(ConfigJsondata["hyperparameters"]["iter"]),(ConfigJsondata["hyperparameters"]["learningRate"]))
vocabWordsDictArr = np.asarray(vocabWordsDict).reshape(len(vocabWordsDict),1)
print "Finished"
res = np.concatenate((vocabWordsDictArr, W),axis=1)
np.savetxt('vectors.txt', res, delimiter=" ", fmt="%s")

