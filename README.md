# Classifying-if-tweet-is-regarding-disaster-or-not-using-RNN-and-NLP

The project uses embedded learning technique where the final output is taken as mode of output of 3 different deep learning models.

Since the input to the model is the text of the tweet, hence concepts of Natural Language Processing (NLP) have to be applied for data preprocessing 

The following preprocessing steps have been used: 
        
        •	• Removal of URLs 
        •	• Removal of tags 
        •	• Removal of Emojis 
        •	• Replacement of common abbreviations with full words of same exact meaning 
        •	• Removal of punctuations, numbers and extra spaces 
        •	• Removal of single character words (like a, I, etc.) 
        •	• Converting all the words to lower alphabets 

After the preprocessing of sentences and getting reduced sentences the input data is to be converted in a bag of words and to be tokenized to matrix unique numbers where each number represents a unique word in the bag of words. This is done using Tokenizer() function included in tensorflow
Since the structure of sentences that is arrangements of words in the sentence will matter significantly in determining the correct output, hence  deep learning involving RNN (LSTM) and ANN using tensorflow backend is used. 

Used for word embeddings and easy mapping of words to vectors and get similar words to the words used in training; as the first layer of each model used. The embedding layers gives its output as input to the first RNN layer in all models. 
In order to make embedding matrix for the embedding layer, pretrained GloVe pretrained word vectors, glove.6B.300d.txt is used

The specification of the 3 models used in ensemble modelling are: 
    
    • Model 1: Embedding layer, followed by a single LSTM layer and the output layer
    • Model 2: Embedding layer, followed by a single LSTM and ANN layer and the output layer 
    • Model 3: Embedding layer, followed by 2 LSTM and a single ANN layer and the output layer 

Each model is trained with same input training data for 15 epochs in batch size of 128 on TPU of Google Collab. 
Optimizer used: Adam 
Activation function: 
    
    for output layer: sigmoid 
    for hidden ANN layer: relu 
Metrics: accuracy 
Loss function: binary cross entropy
