import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pprint, time
from sklearn.metrics import fbeta_score
import seaborn as sns

def fiveFolds(data):
    n=len(data)
    n_split = n//5
    # random.shuffle(data)
    folds=[]
    train_folds = []
    for s in range(0,n,n_split):
        end=min(n,s+n_split)
        fold=data[s:end]
        train_fold = data[:s]
        train_fold.extend(data[end:])
        folds.append(fold)
        train_folds.append(train_fold)
    return [folds,train_folds]

def transition_probability(tag1,tag2,t):
    tags=[pair[1] for pair in t]
    tagc1=tags.count(tag1)
    tagc2=0
    tagsn=np.array(tags)
    tagsc=np.where(tagsn==tag1)[0]
    for i in tagsc:
        if(i!=len(tags)-1 and tags[i+1]==tag2):
            tagc2+=1
    return float(tagc2/tagc1)

def emission_probability(word,tag,t):
    tags = freq[0]
    tagsandwords = freq[2]
    return float (tagsandwords[(word,tag)]/tags[tag])

def Viterbi(test_set,train_set):
    state=[]
    # tags=list(set(p[1] for p in train_set))
    tags=list(freq[0].keys())
    tag_matrix = prob_matrices[0]
    emission_matrix = prob_matrices[1]
    for k,w in enumerate(test_set):
        p=[]
        # tags =list[freq[0].keys()];
        # words = freq[1].keys();
        for tag in tags:
            # print(k)
            if(k==0):
                t_p=tag_matrix[tags.index('.'),tags.index(tag)]
                # t_p=trans_df.loc['.',tag]
            else:
                t_p=tag_matrix[tags.index(state[-1]),tags.index(tag)]
                # t_p=trans_df.loc[state[-1],tag]
            try: 
                e_p=emission_df.loc[test_set[k],tag]
                # e_p=emission_matrix[test_set[k],tag]
            except KeyError:
                # if tag=='X':
                e_p= 1
                # else:
                    # e_p=0

            s_p=t_p*e_p
            p.append(s_p)
        # print(p)    
        max_p=max(p)
        state_max=tags[p.index(max_p)]
        state.append(state_max)  
    return list(zip(test_set,state))

def find_freq(train_words):
  tags = nltk.FreqDist(j for (i,j) in train_words)
  words=nltk.FreqDist(i for (i,j) in train_words)
  words_tags=nltk.FreqDist((i,j) for (i,j) in train_words)
  return ([tags,words,words_tags])

def find_probs(tags_list, words_list,train_words):
  trans_mat=np.zeros((len(tags_list),len(tags_list)),dtype='float32')
  emiss_mat=np.zeros((len(words_list), len(tags_list)), dtype='float32')
  for i, t1 in enumerate(list(tags_list)):
      for j, t2 in enumerate(list(tags_list)): 
          trans_mat[i,j] = transition_probability(t1,t2,train_words)  
  for i, t1 in enumerate(list(words_list)):
     for j, t2 in enumerate(list(tags_list)): 
        emiss_mat[i,j] = emission_probability(t1, t2,train_words) 
  return([trans_mat,emiss_mat])

nltk.download('brown')
from nltk.corpus import brown
nltk.download('universal_tagset')
data = list(nltk.corpus.brown.tagged_sents(tagset='universal'))
result = fiveFolds(data) # returning testdataset and traindataset
folds = result[0]  
train_folds = result[1]
accuracy = []
actual=[]
predicted=[]
f1=[[]]
fhalf=[[]]
f2=[[]]
r=[[]]
p=[[]]
for i in range(len(folds)):
    train_set = train_folds[i]
    test_set = folds[i]
    train_words=[tup for sent in train_set for tup in sent]
    test_words=[tup for sent in test_set for tup in sent]
    freq = find_freq(train_words) #freq[0] = tags_df, freq[1] = words_df , freq[2] = words+tags_df
    #to get tag_set = freq[0].keys() whose length here is 12
    # to get word_set = freq[1].keys()
    #to get tag_word _set = freq[2].keys()
    # print((freq[2].keys()))
    prob_matrices = find_probs(freq[0].keys(),freq[1].keys(),train_words) # return transmission_mat = prob_matrixces[0], emission_mat = [1]
    trans_df = pd.DataFrame(prob_matrices[0],columns=list(freq[0].keys()),index=list(freq[0].keys()))
    emission_df = pd.DataFrame(prob_matrices[1],columns=list(freq[0].keys()),index=list(freq[1].keys()))
    test_data = [tup[0] for sent in test_set for tup in sent] #untagged test data
    start = time.time()
    tagged_seq = Viterbi(test_data,train_set)
    end = time.time()
    difference = end-start
    #print("Time taken in seconds: ", difference)
    check = [i for i, j in zip(tagged_seq, test_words) if i == j] 
    per_pos_acc=[]
    # print(tagged_seq[1])
    # print(test_set[0][1])
    tagged_vit=[i[1] for i in tagged_seq]
    tagged_act = [i[1] for i in test_words]
    accurac = len(check)/len(tagged_seq)
    #print(accurac*100)
    accuracy.append(accurac*100)
    actual.extend(tagged_act)
    predicted.extend(tagged_vit)

print("\n\n accuracy\n\n", accuracy)

con=confusion_matrix(actual,predicted,labels=list(freq[0].keys()))
row=np.sum(con,axis=1)#row sum
col = np.sum(con,axis=0) # col sum
tags = list(freq[0].keys())
j = 0
for i in enumerate(tags):
  print("accuracy per ",i[1]," is :", con[j][j]/row[j])
  j=j+1

c_df=pd.DataFrame(con,columns=list(freq[0].keys()),index=list(freq[0].keys()))
# c_df each row consists of the tested tags for a given tag(actual)
print("\n \n \n ---------------------------- Confusion matrix ------------------------------------------------\n\n")
print(c_df)
a=precision_recall_fscore_support(actual,predicted,labels=list(freq[0].keys()),beta=1)
recall=a[0]
precision=a[1]
f1_score=a[2]
fhalf_score=fbeta_score(actual,predicted,average=None,beta=0.5)
f2_score=fbeta_score(actual,predicted,average=None,beta=2)

print("\n\n         ----------------heatmap----------\n\n")
cmap = sns.cm.rocket_r
sns.heatmap(c_df,cmap=cmap)
