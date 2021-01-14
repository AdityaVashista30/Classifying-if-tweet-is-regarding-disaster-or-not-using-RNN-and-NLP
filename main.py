# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:57:39 2020

@author: aditya
"""
import pandas as pd
import numpy as np
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from tensorflow.keras.optimizers import Adam

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

print(train.size,train.shape)
print(test.size,test.shape)

print(train.head(10))

print(len(train[train['target']==1]))
print(len(train[train['target']==0]))

#PREPROCESSING
def removeURL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def removeTags(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


def onlyWords(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def removeEmoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def fullAbb(text):
    abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "wyd":"what are you doing",
    "doin":"doing",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"}
    l=text.split()
    for i in range(len(l)):
        if l[i].lower() in abbreviations.keys():
            l[i]=abbreviations[l[i].lower()]
    text=' '.join(l)
    return text


def preProcessing(x):

    for i in range(len(x)):
        x[i]=removeURL(x[i])
        x[i]=removeTags(x[i])
        x[i]=removeEmoji(x[i])
        x[i]=fullAbb(x[i])
        x[i]=onlyWords(x[i])
        x[i]=x[i].lower()
    return x



from datetime import datetime
x,y=train.iloc[:,3].values,train.iloc[:,4]
x_test=test.iloc[:,3].values
print(datetime.now().time())
x=preProcessing(x)
x_test=preProcessing(x_test)
print(datetime.now().time())

words,maxwords,maxlen=0,0,0
u=[]
for i in x:
    maxwords=maxwords if maxwords>len(i) else len(i)
    l=i.split()
    maxlen=maxlen if maxlen>len(l) else len(l)
    for j in l:
        if j not in u:
            u.append(j)
            words+=1
print(words)
print(maxwords)
print(maxlen)
    

tokenizer=Tokenizer()
tokenizer.fit_on_texts(x)
x= tokenizer.texts_to_sequences(x)
x_test= tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
maxlen = 65

x = pad_sequences(x, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

#GLOVE FILE 1:
embeddings_dictionary = dict()
glove_file1 = open('glove.6B.300d.txt', encoding="utf8")


for line in glove_file1:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file1.close()

embedding_matrix = np.zeros((vocab_size, 300))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)
#MODEL 1
model1 = Sequential()
model1.add(embedding_layer)
model1.add(LSTM(128))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.fit(x,y,epochs=15,batch_size=128)

ypred1=model1.predict(x_test)
ypred1.resize(len(ypred1))
ypred1=(ypred1>0.5)
sub=pd.read_csv("sample_submission.csv")
for i in range(len(ypred1)):
    sub['target'][i]=int(ypred1[i])
sub.to_csv('submissionM1.csv',index=False)
model1.save("model1I.h5")

#model2
model2=Sequential()
#embedding=Embedding(vocab_size,300,embeddings_initializer=Constant(embedding_matrix),input_length=maxlen,trainable=False)

embedding= Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)

model2.add(embedding)
model2.add(SpatialDropout1D(0.2))
model2.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model2.add(Dense(1, activation='sigmoid'))
optimzer=Adam(learning_rate=1e-5)
model2.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model2.fit(x,y,epochs=15,batch_size=128)

ypred2=model2.predict(x_test)
ypred2.resize(len(ypred1))
ypred2=(ypred2>0.5)
for i in range(len(ypred1)):
    sub['target'][i]=int(ypred2[i])
sub.to_csv('submissionM1.csv',index=False)
model2.save("model2I.h5")
#model 3
model3=Sequential()
embedding= Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model3.add(embedding)
model3.add(LSTM(128, return_sequences = True))
model3.add(LSTM(128, dropout=0.2,  return_sequences = True))
model3.add(LSTM(128, dropout=0.2,  return_sequences = True))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model3.fit(x,y,epochs=6,batch_size=128)

ypred3=model3.predict(x_test)
ypred3.resize(len(ypred3))
ypred3=(ypred3>0.5)
for i in range(len(ypred3)):
    sub['target'][i]=int(ypred3[i])
sub.to_csv('submissionM3.csv',index=False)
model3.save("model3I.h5")

#model 4:
model4=Sequential()
embedding= Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model4.add(embedding)
model4.add(SpatialDropout1D(0.2))
model4.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
model4.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
model4.add(Dense(units=10,activation="relu"))
model4.add(Dense(units=15,activation="relu"))
model4.add(Dense(units=10,activation="relu"))
model4.add(Dense(1, activation='sigmoid'))
optimzer=Adam(learning_rate=1e-5)
model4.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model4.fit(x,y,epochs=10,batch_size=128)

ypred4=model4.predict(x_test)
ypred4.resize(len(ypred4))
ypred4=(ypred4>0.5)
for i in range(len(ypred4)):
    sub['target'][i]=int(ypred4[i])
sub.to_csv('submissionM4.csv',index=False)
model4.save("model4I.h5")

#model 5
model5=Sequential()
embedding= Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model5.add(embedding)
model5.add(SpatialDropout1D(0.2))
model5.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model5.add(Dense(units=10,activation="relu"))
model5.add(Dense(1, activation='sigmoid'))
model5.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model5.fit(x,y,epochs=15,batch_size=128)

ypred5=model5.predict(x_test)
ypred5.resize(len(ypred5))
ypred5=(ypred5>0.5)
for i in range(len(ypred5)):
    sub['target'][i]=int(ypred5[i])
sub.to_csv('submissionM5.csv',index=False)
model5.save("model5I.h5")

#model 6
model6=Sequential()
embedding= Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model6.add(embedding)
model6.add(SpatialDropout1D(0.2))
model6.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

model6.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model6.add(Dense(units=10,activation="relu"))
model6.add(Dense(1, activation='sigmoid'))
model6.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model6.fit(x,y,epochs=15,batch_size=128)

ypred6=model6.predict(x_test)
ypred6.resize(len(ypred6))
ypred6=(ypred6>0.5)
for i in range(len(ypred6)):
    sub['target'][i]=int(ypred6[i])
sub.to_csv('submissionM6.csv',index=False)
model6.save("model6I.h5")

#EMBEDDED MODEL OF 1,5 & 6
from statistics import mode 
for i in range(len(ypred6)):
    sub['target'][i]=int(mode([ypred6[i],ypred5[i],ypred1[i]]))
sub.to_csv('submissionEm.csv',index=False)
