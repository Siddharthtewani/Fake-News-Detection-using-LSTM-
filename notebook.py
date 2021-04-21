#%%
import pandas as pd 
import numpy as np 
import time , glob , re ,os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense , Dropout
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

print("Done")
#%%
df=pd.read_csv(r"C:\Users\siddh\Desktop\Projects\fake_news_classifier\fake-news\train.csv")

df.head()
# %%
df.isnull().sum()
#%%
df.dropna(inplace=True)
# %%
df.shape
# %%
x=df.drop(["label"],axis=1)
y=df["label"]
# %%
print("X shape--->",x.shape)
print("Y shape--->",y.shape)
# %%
voc_size=5000
# %%
messages=x.copy()
# %%
messages.reset_index(inplace=True)
# %%

# %%            Data Preprocessing 
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    print(i)
    review=re.sub('[\W]',' ',messages["title"][i])
    review=review.lower()
    review=review.split()

    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=" ".join(review)
    corpus.append(review)

# %%
corpus
# %%
one_hot_rep=[one_hot(word,voc_size) for word in corpus]
one_hot_rep
# %%
sen_seq=20
pad_sent=pad_sequences(one_hot_rep,padding='pre',maxlen=sen_seq)
pad_sent.shape
# %%
pad_sent
# %%
def build_model(embedding_features):
    model=Sequential()
    model.add(Embedding(voc_size,embedding_features,input_length=sen_seq))
    model.add(LSTM(100))

    model.add(Dense(100,activation="sigmoid"))
    model.add(Dropout(0.2))

    model.add(Dense(100,activation="sigmoid"))
    model.add(Dropout(0.2))

    model.add(Dense(1,activation="sigmoid"))
    print(model.summary())
    return model
# %%
model= build_model(embedding_features=50)
# %%
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
# %%
# %%
x_final=np.array(pad_sent)
y_final=np.array(y)
# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.25,random_state=100)

# %%
print("X train Size--->",x_train.shape)
print("y train Size--->",y_train.shape)
print("X test Size--->",x_test.shape)
print("y test Size--->",y_test.shape)
# %%
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=10)
# %%
# %%
model.evaluate(x_test,y_test)
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

