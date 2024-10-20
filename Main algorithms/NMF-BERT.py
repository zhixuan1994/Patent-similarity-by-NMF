# %%
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# %%
from nltk.stem.porter import *


import tensorflow as tf
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.decomposition import NMF
from numpy import linalg as LA

from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# %%
def read_all_line(file):
    with open(file, 'rt', encoding="utf8") as fd:
        first_line = fd.readlines()
    return first_line

def dict_update(fold_dir,output_file):
    txt_dict = os.listdir(fold_dir)
    out_txt = []
    for file in txt_dict:
        file_dir = fold_dir + file
        temp_txt = read_all_line(file_dir)
        out_txt += temp_txt
    out_txt = list(set(out_txt))

    with open(output_file, 'w', encoding="utf8") as output:
        for txt_data in out_txt:
            output.write(txt_data)

def pre_token(txt_file, stop_words):
    txt_pre = re.sub('\d','', txt_file)
    txt_pre = re.sub(r"[^a-zA-Z]+", ' ', txt_file)
    return [w for w in word_tokenize(txt_pre) if w not in stop_words]

def vectorize(sentences, BERT_model):
    return np.array(BERT_model.encode(sentences, show_progress_bar=False))


# %%
stop_words = open('stop_dict_ai.txt', 'r').read()
stop_words = stop_words.split('\n')
BERT_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# %% [markdown]
# ### Preprocessing
# 
# Keep the IPC where amount of patends greater than 300

# %%
class Auto_encoder(Model):
  def __init__(self, dim_out, dim_encod):
    super(Auto_encoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(dim_out),
      layers.Dense(512, activation="tanh"),
      layers.Dense(256, activation="tanh"),
      layers.Dense(128, activation="tanh"),
      layers.Dense(dim_encod)])

    self.decoder = tf.keras.Sequential([
      layers.Dense(dim_encod),
      layers.Dense(128, activation="sigmoid"),
      layers.Dense(256, activation="sigmoid"),
      layers.Dense(512, activation="sigmoid"),
      layers.Dense(dim_out, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# %%
IPC_train = pd.read_csv('IPC_data_ai_train.csv')
BERT_train = pd.read_csv('BERT_ai.csv')
IPC_test = pd.read_csv('IPC_data_ai_test.csv')

# %%
def pre_pro_all(BERT_IPC, IPC_name, IPC_data):
    sentens_ID = IPC_data[IPC_data[IPC_name]==1]['ID']
    sentences = []
    for id in sentens_ID:
        temp_vec = BERT_IPC[BERT_IPC['ID']==id]
        temp_vec = temp_vec.drop('ID', axis=1)
        if len(temp_vec)>1:
            print('Wrong, Same ID')
        sentences.append(temp_vec)
    sentences = np.asarray(sentences)
    return np.mean(sentences, axis=0)

def cos_sim(p1, p2):
    p1 = np.array(p1).reshape(1,-1)
    p2 = np.array(p2).reshape(-1,1)
    return p1.dot(p2)/(LA.norm(p1)*LA.norm(p2))

def sim_res(p1, rest):
    sent_sim_res = []
    for one_ex in rest:
        sent_sim_res.append(cos_sim(p1, one_ex)[0][0])
    return np.asarray(sent_sim_res)

def main_process(IPC_train, BERT_train, BERT_test, ID_test):

    Train_auto_data = BERT_train.drop('ID', axis=1)
    Train_auto_data = Train_auto_data.to_numpy()
    train_sc = MinMaxScaler()
    Train_auto_data = train_sc.fit_transform(Train_auto_data)
    np.random.shuffle(Train_auto_data)
    X_train_auto = Train_auto_data[: int(len(Train_auto_data)*0.7)]
    X_test_auto = Train_auto_data[int(len(Train_auto_data)*0.7):]

    autoencoder = Auto_encoder(X_train_auto.shape[1], 32)
    # For MSE
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    # For BCE
    # autoencoder.compile(optimizer='adam', loss=losses.BinaryCrossentropy())
    autoencoder.fit(X_train_auto, X_train_auto,
                epochs=30,
                shuffle=True,
                validation_data=(X_test_auto, X_test_auto), verbose=None)
    
    BERT_train_data = autoencoder.predict(Train_auto_data, verbose=None)
    BERT_train_data = pd.DataFrame(BERT_train_data)
    BERT_train_data['ID'] = BERT_train['ID']

    H_ini_mean = []
    for cols in IPC_train.columns.drop(['ID','abstract']):
        col_temp = pre_pro_all(BERT_train_data, cols, IPC_train)[0]
        H_ini_mean.append(col_temp)
    H_ini_mean = np.asarray(H_ini_mean)

    # print(BERT_train_data)
    A_data = BERT_train_data.drop('ID', axis=1)
    A_data = A_data.to_numpy()
    W_ini = IPC_train.drop(['ID','abstract'], axis=1)
    W_ini = W_ini.to_numpy()

    A_data = A_data.astype('float64')
    W_ini = W_ini.astype('float64')
    H_ini_mean = H_ini_mean.astype('float64')

    A_data = np.ascontiguousarray(A_data)
    W_ini = np.ascontiguousarray(W_ini)
    H_ini_mean = np.ascontiguousarray(H_ini_mean)

    NMF_md = NMF(n_components = W_ini.shape[1],init='custom', max_iter=30000)
    W_out_mean = NMF_md.fit_transform(A_data, W = W_ini, H= H_ini_mean)
    H_out_mean = NMF_md.components_

    BERT_test_temp = train_sc.transform(BERT_test)
    IPC_test = autoencoder.predict(BERT_test_temp, verbose=None)
    IPC_test = IPC_test.astype('float')
    NMF_test = NMF_md.transform(IPC_test)

    sim_result = sim_res(NMF_test[0], NMF_test)
    res_dict = dict([(ID_test[k], [sim_result[k]]) for k in range(len(ID_test))])
    return res_dict

# %%
test_data = IPC_test
sentences = test_data['abstract']
token,sent_array = [],[]

for i,j in enumerate(sentences):
    temp = pre_token(j, stop_words)
    token.append(' '.join(np.array(temp)))
test_data['token'] = token
test_data = test_data.dropna()
test_data = test_data.loc[test_data['token']!='',:].reset_index(drop=True)

token_list = test_data['token']
for sent in tqdm(token_list):
    temp = vectorize(sent, BERT_model)
    sent_array.append(temp)
sent_array = np.asarray(sent_array)

# %%
res_df = pd.DataFrame(columns=test_data['ID'])
for i in tqdm(range(100)):
    res_temp_dict = main_process(IPC_train, BERT_train, sent_array, test_data['ID'])
    res_temp_df = pd.DataFrame(res_temp_dict)
    res_df = pd.concat([res_df, res_temp_df], ignore_index=True)

# %%
res_df.to_csv('test_results.csv',index=False)

# %%



