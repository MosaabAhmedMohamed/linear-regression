from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features=[8,7]


"""
for i in dataset:
    for ii in dataset[i]:
        [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]

plt.scatter(new_features[0],new_features[1])
plt.show()
"""

def K_nearest_neighbors(data,predict,k=3):
    if len (data) >= k:
        Warning.warn('K is to a values less than total voting groups !')

    distances = []
    for group in data:
     for features in data[group]:
        euclidean_distance = np.linalg.norm((features)-np.array(predict))
        distances.append([euclidean_distance,group])

     votes = [i[1]for i in sorted(distances)[:k]]
     vote_result = Counter(votes).most_common(1)[0][0]
     confidance = Counter(votes).most_common(1)[0][1]/k
     #print(vote_result+" "+confidance)

    return vote_result , confidance

accurcies=[]

for i in range(25):


 df=pd.read_csv('breast-cancer-wisconsin.data')
 df.replace('?',-99999,inplace=True)

 df=df.iloc[:,1:11]
 full_data=df.astype(float).values.tolist()
 X=df.iloc[:,1:10]
 Y=df.iloc[:,10:11]


 random.shuffle(full_data)

 test_size=0.2
 train_set={2:[],4:[]}
 test_set={2:[],4:[]}
 train_data=full_data[:-int(test_size*len(full_data))]
 test_data=full_data[:-int(test_size*len(full_data)):]

 for i in train_data:
    train_set[i[-1]].append(i[:-1])

 for i in test_data:
    test_set[i[-1]].append(i[:-1])


 correct=0
 total=0

 for group in test_set:
    for data in test_set[group]:
        vote,confidance=K_nearest_neighbors(train_set,data,k=5)
        if group==vote:
            correct+=1
            total += 1

 #print("Accuracy :",correct/total)
 accurcies.append(correct/total)
print(sum(accurcies)/len(accurcies))




