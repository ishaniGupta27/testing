import numpy as np
import json
import zipfile
import sys
import math 
with open('config.json') as json_data_file:
    ConfigJsondata = json.load(json_data_file)

#------------------------------------------------------------------------------------------##
#------------------------------BUILDING THE VOCAB DICT start-------------------------------------##
#------------------------------------------------------------------------------------------##
globIndx=0
globCount=170
globInitPara=0
globalStart=1
globStepSize=100000
#builing vocab --> setting vocab size and making hot vectors !
with open("vocab.txt","r") as f:
    data = f.readlines()
vocab_file  = open("vocab.txt", "r") #just wanna read the file
word2int={}#hash map
int2word={}#hash map

vocabWordsDict=[] #Array 
vocabWordsDictCounter={}#dixtionary 
for word in data:
       if word!='.': #'.' is not to be considered as a word
           vocabWordsDict.append(word.split('\n')[0])
           vocabWordsDictCounter[word.split('\n')[0]]=0;


vocabSize=len(vocabWordsDict)#How many words we have in our vocablury

for i,word in enumerate(vocabWordsDict):
   
   word2int[word] = i
   int2word[i] = word 
print "The Vocab Size is :::"
print vocabSize
#------------------------------------------------------------------------------------------##
#------------------------------BUILDING THE VOCAB DICT end---------------------------------##
#------------------------------------------------------------------------------------------##




#------------------------------------------------------------------------------------------##
#------------------------------Unigram starts----------------------------------------------##
#------------------------------------------------------------------------------------------##
class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab,vocabDictCount):
        vocab_size = len(vocab)
        power = 0.75
        norm=0
        for k, v in vocabDictCount.items():
            norm +=(math.pow(v, power)) # Normalizing constant
        

        #print'----'
        #print norm
        table_size = 1000 # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        #print 'Filling unigram table'
        p = 0 # Cumulative probability
        i = 0
        for j, k in enumerate(vocabDictCount):
            v=vocabDictCount[k]
            #print v
            #print norm
            p += float(math.pow(v, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

#------------------------------------------------------------------------------------------##
#------------------------------Unigram ends----------------------------------------------##
#------------------------------------------------------------------------------------------##

#---------------------------------HELPER FUNCTIONS-----------------------------------------##
def to_one_hot(data_point_index, vocab_size):
    """
    Computing the on_hot_vector for the word.

    Argument:
    data_point_index -- int of the word to be represented by one hot vector
    vocab_size --- Vocabulury Size
    Returns:
    temp -- a 1*v vector
     """
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def hot_to_number(data_point_index):
    res=0
    it = np.nditer(data_point_index, flags=['f_index'])
    while not it.finished:
        if it[0] == 1.0:
            res=it.index
            break
        it.iternext()
        
    return res

def hot_to_number_new(data_point_index):
    try:
        assert (len(data_point_index) == vocabSize) 
    except:
        print data_point_index.shape
    res=0;
    for i,val in enumerate(data_point_index):
        if val.all() == 1.0:
            res=i;
    assert (res<=vocabSize)
    return res

#------------------------------------------------------------------------------------------##

#------------------------------------------------------------------------------------------##
#------------------------------LOADING TEST DATA start-------------------------------------##
#------------------------------------------------------------------------------------------##

corpus_raw=""
corpus_size=0
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

corpus_raw = read_data('text8.zip')
corpus_size=len(corpus_raw)
sys.stdout.write("\n Our Corpus size is-------------\n")
sys.stdout.write(str(len(corpus_raw))) #len(data)
sys.stdout.write("\n--------------------------------\n")
#print (corpus_raw[1:10]) #this data is clean data..no fulls ops nohhing

#corpus_raw = fileData.readlines()
#corpus_raw="hot king  king .  king  terror . sex king   queen"
#TO CHECK :print word2int['zoo'] #not in order...does it matter??
#converting to lower case
#corpus_raw=corpus_raw.lower()
#handling multiple sentences
#raw_sentences=corpus_raw.split('.')##Split sentences to test and train here ..maybe??
#sentDict=[] #these are the words in sentences
#for sen in raw_sentences:
    #sentDict.append(sen.split())
#print sentDict[1]
#Lets generate the training set according to window
#data=[] #it will have one center word and its neighbours for many center words

def fillCorpus(globalStart,globUpto):
    x_train =[]
    y_train=[]
    data = [] 
    x_train_real = [] # input word--Center word in our case
    y_train_real = [] # output word---> nb word ! the one forming a tuple
    #print "-----------Length of the tuples----------------------"
    WINDOW_SIZE = int(ConfigJsondata["hyperparameters"]["window_size"])
	#for sentence in sentDict: #pick a sentence at a time
    for word_index, word in enumerate(corpus_raw[globalStart:globUpto]):
	    #if word_index%100000 == 0:
	        #print word
	    if(word_index>=WINDOW_SIZE & word_index<(corpus_size-10)):
	        counter=0;
	        while counter<WINDOW_SIZE:
	            if word in vocabWordsDict:
	                vocabWordsDictCounter[word]=vocabWordsDictCounter[word]+1
	                #print 'yoooooooo'
	                #print vocabWordsDictCounter[word]
	                if corpus_raw[word_index-(counter+1)] in vocabWordsDict:
	                    #print "------------------"
	                    #print corpus_raw[word_index-(counter)]
	                    vocabWordsDictCounter[corpus_raw[word_index-(counter+1)]]=vocabWordsDictCounter[corpus_raw[word_index-(counter+1)]]+1
	                    data.append([word, corpus_raw[word_index-(counter+1)]])# 2-0-1=1,2-1-1=0
	                #if((word_index+counter+2)%10000==0):
	                    #sys.stdout.write("\n -------------------------------- \n")
	                    #sys.stdout.write(str(word_index+counter+2)) #len(data)
	                    #sys.stdout.write("\n --------------------------------\n")
	                if (word_index+counter+2)<(corpus_size-2):
	                    if corpus_raw[word_index+counter+2] in vocabWordsDict:
	                        #print "------------------"
	                        #print corpus_raw[word_index+counter+2]
	                        vocabWordsDictCounter[corpus_raw[word_index+counter+2]]=vocabWordsDictCounter[corpus_raw[word_index+counter+2]]+1
	                        data.append([word, corpus_raw[word_index+counter+2]])# 2+0+1=3,2+1+1=4
	                #else:
	                   # data.append([word, "markedUn"])# marking unknown 
	                    
	            counter=counter+1


    sys.stdout.write("\n ------------The Tuple size-------------------- \n")
    sys.stdout.write(str(len(data))) #len(data)
    sys.stdout.write("\n --------------------------------\n")
    for indx,data_word in enumerate(data):
	    #print data_word
	    globIndx=indx
	    if indx%100000==0:
	        print '-----------'
	        print indx
	    x_train_real.append(word2int[ data_word[0]])
	    y_train_real.append(word2int[ data_word[1]])
	    


	# convert them to numpy arrays
    x_train = np.asarray(x_train_real).reshape(len(x_train_real),1)
	#x_train = np.asarray(x_train_real)
    y_train = np.asarray(y_train_real).reshape(len(y_train_real),1)
    parameters = {"x_train": x_train,
                  "y_train": y_train,}    
    return parameters
    print "Corpus is filled"
#print y_train
#------------------------------------------------------------------------------------------##
#------------------------------DATA TO ONE HOT VECTOR end----------------------------------##
#------------------------------------------------------------------------------------------##

#print 'Shape of training set input'
#print x_train.shape #this is just one word represented by vocab size =V

#print 'Shape of training set output'
#print y_train.shape #this is just one word represented by vocab size =V

#------------------------------------------------------------------------------------------##
#-----------------------------FUNCTIONS START HERE-----------------------------------------##
#------------------------------------------------------------------------------------------##

def softmax_function(x): #X is a matrix 
    """
    Computing the softmax function for each row of the input x to be used for skip gram model.

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
     """
    orig_shape = x.shape
    ndim = x.ndim
    max = np.max(x, axis=ndim-1, keepdims=True)   # max value of array of each row. (M x 1)
    exp = np.exp(x - max)                         # exp of each element. (M x N)
    sum = np.sum(exp, axis=ndim-1, keepdims=True) # sum of each row. (M x 1)
    x = exp / sum                                 # softmax. (M x N)
    assert x.shape == orig_shape
    return x
    
    '''
    # Applying exp() element-wise to x.
    max_x=np.max(x, axis=1)[:,np.newaxis]
    #For numerical stability , finding softmax of (x-maxx)
    np.reshape(max_x,(max_x.shape[0],1))
    #print max_x.shape
    x_exp = np.exp(x-max_x)#this will make all the elemnts change to there exponents

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum ##broadcating make things easier !
    
    return s
   '''

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples) ..... 2*400
    Y -- labels of shape (output size, number of examples)...... 1*400
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    #print X.shape[]
    n_x = X.shape[1] # size of input layer i.e vocabsize for now
    n_h = (ConfigJsondata["hyperparameters"]["dimen"]) #this is size of hidden layer
    n_y = Y.shape[1] # size of output layer i.e vocab size for now
    
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer = 7
    n_h -- size of the hidden layer = 5
    n_y -- size of the output layer = context window * 7
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    W2 -- weight matrix of shape (n_y, n_h)
    """
    W1 = np.random.uniform(low=-0.5/n_h, high=0.5/n_h, size=(n_x, n_h))
    #W1 = np.random.randn(n_x,n_h)*0.08 #this is vocab * dimen==> V*H
    W2 = np.zeros(shape=(n_h, n_y)) #this is dimen * vocab ===> H*V
    #Is there any relation between W1 and W2
   
    
    assert (W1.shape == (n_x, n_h)) 
    assert (W2.shape == (n_h, n_y))
   
    
    parameters = {"W1": W1,
                  "W2": W2,}    
    return parameters

def assign_init_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer = 7
    n_h -- size of the hidden layer = 5
    n_y -- size of the output layer = context window * 7
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    W2 -- weight matrix of shape (n_y, n_h)
    """
    W1 = np.loadtxt("w1.txt")
    #W1 = np.random.randn(n_x,n_h)*0.08 #this is vocab * dimen==> V*H
    W2 = np.loadtxt("w2.txt") #this is dimen * vocab ===> H*V
    #Is there any relation between W1 and W2
   
    
    assert (W1.shape == (n_x, n_h)) 
    assert (W2.shape == (n_h, n_y))
   
    
    parameters = {"W1": W1,
                  "W2": W2,}    
    return parameters
def forward_propagation_negSam(X, parameters,negSampleIndx,Y):
    """
    Argument:
    X -- input data of size (1, VocabSize)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    cache: #to complete
    """
    # Retrieving each parameter from the dictionary "parameters"
    
    W1 = parameters["W1"] 
    W2 = parameters["W2"]
    try:
        assert (Y.shape == (1,vocabSize))
    except:
        print vocabSize
        print Y.shape
    target=hot_to_number(Y)
    #print "Shape of X1"
    #print X.shape  ##this is 1*5

    
    # Implementing Forward Propagation to calculate output vectors
    
    Z1 = np.dot(X,W1)  #X---no. of examples*vocab (1*7), W1:: vocab * dimen = 7*5
    A1 = Z1 ## this is dimen*1
    #print "Shape of A1"
    #print A1.shape  ##this is 1*5

    ##h---------------CONTEXT WINDOW-------------------------------------------------
    Z2_first = np.dot(A1,W2) #W2::: dimension * vocab==> 5*7, A1:: 1* dimension= 1*5
    ##----------------------------------------------------------------------------------
    ##---------------------NEGATIVE SAMPLING--------------------------------------------
    ##---------------------------------------------------------------------------------- 
    ##----------------------------------------------------------------------------------
    #The idea behind negative sampling is to reduce the dimansions of 
    #print "Shape of Z2_first"
    #print Z2_first.shape  ##this is 1*(negative sample+1)
    A2_real = softmax_function(Z2_first)
    #print 'A2_real shape'
    #print A2_real.shape #-----> 1*4894
    #print negSampleIndx
    #print target
    
    allIndex=np.append(negSampleIndx,target)
    #print 'allIndex'
    #print allIndex
    try:
        A2=A2_real[:,allIndex] #A2 is what??
    except:
        print target
        print Y
        print allIndex
    
    #print 'A2 size'
    #print A2.shape
    #print "Shape of A2_first"
    #print A2_first.shape  ##this is 1*7
    '''
    c=0
    val=np.multiply(WINDOW_SIZE,2)
    #the idea is to reduce the
    A2 = np.empty((val, vocabSize))
    while c<val:
        A2[c]=A2_first; #this is the same value repeated all the time ! dimension:: c*vocab!
        c=c+1
    #print "Shape of A2 after loop"
    #print A2.shape  ##this is 4*7--> this is windowsize*vocabsize
    '''


    cache = {"X1":X,
             "Z1": Z1,
             "A1": A1,
             "A2": A2,
             "negInd":allIndex}
    
    return cache

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    cache: #to complete
    """
    # Retrieving each parameter from the dictionary "parameters"
    
    W1 = parameters["W1"] 
    W2 = parameters["W2"]
   
    #print "Shape of X1"
    #print X.shape  ##this is 1*5

    
    # Implementing Forward Propagation to calculate output vectors
    
    Z1 = np.dot(X,W1)  #X---no. of examples*vocab (1*7), W1:: vocab * dimen = 7*5
    A1 = Z1 ## this is dimen*1
    #print "Shape of A1"
    #print A1.shape  ##this is 1*5

    ##h---------------CONTEXT WINDOW-------------------------------------------------
    Z2_first = np.dot(A1,W2) #W2::: dimension * vocab==> 5*7, A1:: 1* dimension= 1*5
    ##----------------------------------------------------------------------------------
    ##---------------------NEGATIVE SAMPLING--------------------------------------------
    ##---------------------------------------------------------------------------------- 
    ##----------------------------------------------------------------------------------
    #The idea behind negative sampling is to reduce the dimansions of 
    #print "Shape of Z2_first"
    #print Z2_first.shape  ##this is 1*5
    A2 = softmax_function(Z2_first)
    #print "Shape of A2_first"
    #print A2_first.shape  ##this is 1*7
    '''
    c=0
    val=np.multiply(WINDOW_SIZE,2)
    #the idea is to reduce the
    A2 = np.empty((val, vocabSize))
    while c<val:
        A2[c]=A2_first; #this is the same value repeated all the time ! dimension:: c*vocab!
        c=c+1
    #print "Shape of A2 after loop"
    #print A2.shape  ##this is 4*7--> this is windowsize*vocabsize
    '''


    cache = {"X1":X,
             "Z1": Z1,
             "A1": A1,
             "A2": A2}
    
    return cache


def stg_update_parameters(parameters,errorArr,cache,learning_rate,dimen):
    '''
      errorArr: c*vocab
      W2: d*vocab-->(for w1 take transpose)
      x--1*vocab



    '''
    i=0
    j=0
    W1=parameters["W1"]
    W2=parameters["W2"]
    A1=cache["A1"]
    X1=cache["X1"]
    negInd=cache["negInd"]
    #print 'Error Arr shape'
    #print errorArr.shape ##---> (neg sample+1 *1)

    #print 'A1'
    #print A1.shape
    ##similar to be done for W1
    '''
    while i<dimen :
        while j< vocabSize:
            cLay=0;
            sum=0;
            while cLay<(WINDOW_SIZE*2):
                sum=sum+errorArr[cLay,j]*A1[:,i]
                cLay=cLay+1
            W2[i][j]=W2[i][j]-(learning_rate*sum)
            j=j+1
        i=i+1
    '''
    #vectorizing
    '''
    c=0
    val=np.multiply(WINDOW_SIZE,2)
    A1_moif = np.empty((val, dimen))
    while c<val:
        A1_moif[c]=A1;
        c=c+1
    '''
    #print "error shape is"
    #print errorArr.shape #--->this is 1* (negasamp+1)

    #print "A1 is"
    #print A1.shape #--->this is(1*dimen)

    #print "W2 && negInd is"
    #print negInd
    #print W2[:,negInd].shape #--->this is 4*7
    #W2_new=W2[:,negInd] # this dimen * (negsample+1)
    W2[:,negInd]=W2[:,negInd]-np.transpose((learning_rate)*(np.dot(np.transpose(errorArr),A1))) #after - transpose((negasample+1) * dimen)
    #print 'W2 shape'
    #print W2.shape
    ##similar to be done for W2
    '''
    c=0
    X1_moif = np.empty((val, vocabSize))
    while c<val:
        X1_moif[c]=X1;
        c=c+1
    '''
    #This is for which this whole thing is done for !
    '''
    print '------------------------------------------'
    print '----------------here 1--------------------------'
    print hot_to_number(X1)
    print '------------------------------------------'
    print '----------------here 1--------------------------'
    print W1[hot_to_number(X1-1),:] 
    print W1[hot_to_number(X1),:] #Sanity Check ! there should be change in central word!
    print '------------------------------------------'
    print '------------------------------------------'
    '''
    W1=W1-np.transpose((learning_rate)*(np.dot(np.transpose(np.dot(errorArr,np.transpose(W2[:,negInd]))),X1)))
    '''
    print '------------------------------------------'
    print '---------------here 2---------------------------'
    print W1[hot_to_number(X1-1),:] #Sanity check
    print W1[hot_to_number(X1),:] #Sanity check
    print '------------------------------------------'
    print '------------------------------------------'
    
    '''
    parameters = {"W1": W1,
                  "W2": W2}
    return parameters
    
#------------------------------------------------------------------------------------------##
#-----------------------------FUNCTIONS END HERE-----------------------------------------##
#------------------------------------------------------------------------------------------##   
'''

def skipgram_model_dummy(X, Y, n_h):
    
    n_x = layer_sizes(X, Y)[0] #==vocabsize=7
    n_y = layer_sizes(X, Y)[2] #==vocabsize=7
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    cache = forward_propagation(X, parameters)
    #this is getting hardcoded
    

    ##this is part of backprop
    target = np.empty(((WINDOW_SIZE*2), vocabSize))
     #hardcoded ! ##need to be changed 
    target[0]=to_one_hot(word2int["is"],vocabSize) #expected Output
    target[1]=to_one_hot(word2int["the"],vocabSize)
    target[2]=to_one_hot(word2int["is"],vocabSize) #expected Output
    target[3]=to_one_hot(word2int["the"],vocabSize)
    print "target shape is "
    print target.shape


    errorArr=target-cache["A2"]
    print "error shape is"
    print errorArr.shape #--->this is 4*7

    print stg_update_parameters(parameters,errorArr,cache,0.1,n_h)

    Questions to ask:::

    1. How to decide context window? How to decide hidden layer dimensionality?#trial and test
    2. in stocgraddesc: what is hi? is the output of hideen layer?? How to use SQD update(slide 25)--Done !
    3. How the input has to be--> i mena the data?? Should it pick the cebter word and +- window and keep them together.--> Ask Richika
    4. Goal is to get hidden layer parameters for all the center words?--W1 ##Done
    3. how to train?? Split the data an and then cehck teh accuracy !

    '''
#------------------------------------------------------------------------------------------##
#-----------------------------MAIN FUNCTION START HERE-------------------------------------##
#------------------------------------------------------------------------------------------##   
def skipgram_model_loop(X_raw, Y_raw, n_h,globInitPara):
    
    n_x = vocabSize#layer_sizes(X, Y)[0] #==vocabsize=7
    n_y = vocabSize#layer_sizes(X, Y)[2]#==vocabsize=7
    
    
    #print "Shape of X_raw"
    #print X_raw.shape
    if globInitPara ==0:
        parameters = initialize_parameters(n_x, n_h, n_y)
    else:
        parameters=assign_init_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    W2 = parameters["W2"] #stays intact

    #print 'Shape of W2'
    #print W2.shape
    '''
    target = np.empty(((WINDOW_SIZE*2), vocabSize))
     #hardcoded ! ##need to be changed 
    target[0]=to_one_hot(word2int["king"],vocabSize) #expected Output
    target[1]=to_one_hot(word2int["hot"],vocabSize)
    target[2]=to_one_hot(word2int["king"],vocabSize) #expected Output
    target[3]=to_one_hot(word2int["hot"],vocabSize)
    print "target shape is "
    print target.shape
    print '------------------------------------------'
    print '------------------------------------------'
    print parameters["W1"]
    print '------------------------------------------'
    print '------------------------------------------'
    '''
    
    for word_indx in range(len(X_raw)):
        ### START CODE HERE ###
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        # X --> its is no. of center words * vocabsize
        X=to_one_hot(X_raw[[word_indx],:],vocabSize)
        Y=to_one_hot(Y_raw[[word_indx],:],vocabSize)
        X = np.asarray(X).reshape(1,len(X))
        Y= np.asarray(Y).reshape(1,len(Y))
        #print "Shape of X"
        #print X.shape
        if ConfigJsondata["hyperparameters"]["negSampl"] == 1:
                        #Yippee I am going to set a negative sample
                        table = UnigramTable(vocabWordsDict,vocabWordsDictCounter)
                        negSampleSize=ConfigJsondata["hyperparameters"]["negaSampSize"]
                        negSamples=table.sample(negSampleSize)
                        #print negSamples
        cache = forward_propagation_negSam(X, parameters,negSamples,Y)
        target = np.zeros(shape=(1,ConfigJsondata["hyperparameters"]["negaSampSize"]+1))
        target[0,ConfigJsondata["hyperparameters"]["negaSampSize"]]=1;
        #print target
        #print cache["A2"]
        errorArr=target-cache["A2"]
        #print "error shape is"
        #print errorArr.shape #--->this is 4*7
        parameters = stg_update_parameters(parameters,errorArr,cache,(ConfigJsondata["hyperparameters"]["learningRate"]),n_h)
        ##break;
        ### END CODE HERE ###



    return parameters

#------------------------------------------------------------------------------------------##
#-----------------------------MAIN FUNCTION START HERE-------------------------------------##
#------------------------------------------------------------------------------------------##   
    '''
    Questions to ask:::

    1. How to decide context window? How to decide hidden layer dimensionality?learnign rate?
    2. How to use SQD update(slide 25)
    3. How the input has to be--> i menan the data?? Should it pick the cebter word and +- window and keep them together.
    4. Goal is to get hidden layer parameters for all the center words?
    3. how to train??

    '''
while(globCount>1):
    upTo=globalStart+globStepSize-1
    print "taking corpus from %d and ending at %d",globalStart,upTo
    parametersXY=fillCorpus(globalStart,upTo)
    x_train=parametersXY["x_train"]
    y_train=parametersXY["y_train"]
    iter=0
    iterNum=(ConfigJsondata["hyperparameters"]["iter"])
    while iter<iterNum:
        ResWhole=skipgram_model_loop(x_train,y_train,(ConfigJsondata["hyperparameters"]["dimen"]),globInitPara)
        print iter
        globInitPara=1
        globalStart=upTo+1
        np.savetxt('w1.txt', ResWhole["W1"])
        np.savetxt('w2.txt', ResWhole["W2"])
        iter=iter+1
    globCount= globCount-1
  
    

#def mainFunc():
res=skipgram_model_loop(x_train,y_train,(ConfigJsondata["hyperparameters"]["dimen"]),1)["W1"]

#print 'Shape of the final project is:'
#print res.shape
vocabWordsDictArr = np.asarray(vocabWordsDict).reshape(len(vocabWordsDict),1)
print "Finished"
res = np.concatenate((vocabWordsDictArr, res),axis=1)
np.savetxt('vectors.txt', res, delimiter=" ", fmt="%s")

