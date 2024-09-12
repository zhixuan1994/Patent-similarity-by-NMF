# %%
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from nltk.tokenize import word_tokenize
from collections import Counter
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
    # print(sentences)
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

# %% [markdown]
# ### Pure BERT output

# %%
ID_test = test_data['ID']
BERT_res = sim_res(sent_array[0], sent_array)
BERT_sort = np.argsort(BERT_res)[::-1]

# %% [markdown]
# ### BERT(after scaler) + IPC

# %%
train_sc = MinMaxScaler()
test_BERT = train_sc.fit_transform(sent_array)
test_data_IPC = test_data.drop(['abstract','ID','token'], axis=1)
test_BERT = np.concatenate([test_BERT, test_data_IPC], axis=1)

# %%
BERT_IPC_res = sim_res(test_BERT[0], test_BERT)
BERT_IPC_sort = np.argsort(BERT_IPC_res)[::-1]

# %% [markdown]
# ### TF-IDF

# %%
def tf_idf_one(text, stop_words):
    return pre_token(text, stop_words)

def tf_idf_df(data, stop_words):
    key_data0 = tf_idf_one(data[0], stop_words)
    tf_df = dict.fromkeys(key_data0, [1])
    tf_df = pd.DataFrame(tf_df)
    
    for test_d in tqdm(data[1:]):
        temp = tf_idf_one(test_d, stop_words)
        data = Counter(temp)
        temp_df = pd.DataFrame(dict(data), index=[0])
        tf_df = pd.concat([tf_df, temp_df])
    tf_df = tf_df.fillna(0)
    duc_sum = np.asarray(np.sum(tf_df.iloc[:,:-1], axis = 1))
    temp_sum = np.asarray(tf_df.astype(bool).sum(axis=0))
    df_sum = np.asarray(np.log(len(duc_sum)/temp_sum))

    for i in range(len(duc_sum)):
        tf_df.iloc[i, :-1] = tf_df.iloc[i, :-1]/duc_sum[i]
    for i in range(tf_df.shape[1]):
        tf_df.iloc[:, i] = tf_df.iloc[:, i]*df_sum[i]
    return tf_df

# %%
tf_df = tf_idf_df(test_data['abstract'], stop_words)
tf_df = tf_df.fillna(0)
tf_df['ID'] = test_data['ID']
tf_df_noID = tf_df.drop('ID',axis=1)
tf_array = tf_df_noID.to_numpy()

# %%
results_tf = sim_res(tf_array[0], tf_array)
result_sort = np.argsort(results_tf)[::-1]


