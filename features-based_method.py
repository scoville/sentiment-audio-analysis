
# coding: utf-8

# In[18]:


import numpy as np

b =[[1,2], [3,4]]

a= np.array([[5,6],[7,8], [10,11]])
print(a)
print(b)
a = np.append(a,b, axis=0)
print(a)


# In[ ]:


#### Training based on features of audio

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
import random
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier



#Constant
EMOTION_ANNOTATORS = {'anger': 0, 'happiness' : 1, 'sadness' : 2, 'neutral' : 3, 'frustration' : 4, 'excited': 5,
           'fear' : 6,'surprise' : 7,'disgust' : 8, 'other' : 9}

EMOTION = {'ang': 0, 'hap' : 1, 'sad' : 2, 'neu' : 3, 'fru' : 4, 'exc': 5,
           'fea' : 6,'sur' : 7,'dis' : 8, 'oth' : 9, 'xxx':10}


EMOTION = {'hap': 0, 'neu' :1, 'sad' :2}

METHOD = {'audio_feature':0, 'LSTM':1}

#Method for classification
method = METHOD['audio_feature']

#If data is processed and saved into files, just reload, dont need to re-process
isRawDataProcessed = True

#Development mode. Only run with small data.
dev = False



def training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

   

    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    i_fold = 0
    accuracy_train_results = []
    accuracy_valid_results = []

    for train_index, valid_index in kf.split(X_train):
        i_fold = i_fold + 1
        
        x_train_sub, x_valid_sub = X_train[train_index], X_train[valid_index]
        y_train_sub, y_valid_sub = y_train[train_index], y_train[valid_index]
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=30)
        
      
        
        clf = RandomForestClassifier(n_estimators = 300, max_depth=4, random_state=300, class_weight = "balanced")
        clf.fit(x_train_sub, y_train_sub)
        
        score = clf.score(x_train_sub, y_train_sub)
        score1 = clf.score(x_valid_sub, y_valid_sub)
        accuracy_train_results.append(score)
        accuracy_valid_results.append(score1)
        
        print("Score of training set: ", score)
        print("Score of validation set: ", score1)
     
       
    
    avg_accuracy_train_result = np.sum(accuracy_train_results) / len(accuracy_train_results)
    avg_accuracy_valid_result = np.sum(accuracy_valid_results) / len(accuracy_valid_results)
    print("Average accuracy training set, std:", avg_accuracy_train_result, " ",          np.std(accuracy_train_results))
    print("Average accuracy validation set, std:", avg_accuracy_valid_result," ",           np.std(accuracy_valid_results))     
    
    
#     up_sampling_data = []
#     for i in range(0, len(y_train)):
#         if y_train[i] == 1:
#             up_sampling_data.append(X_train[i])
    
#     X_train = np.append(X_train, up_sampling_data, axis = 0)
#     y_train = np.append(y_train, [1] * len(up_sampling_data))
    
    
#     c = list(zip(X_train, y_train))
#     random.shuffle(c)
#     X_train, y_train = zip(*c)
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
    
    clf.fit(X_train, y_train)
   

    predicts = clf.predict(X_test)
    pro = clf.predict_proba(X_test)
 #   print("predicts: ", predicts)
 #   print("prob: ", pro[0])
    score_test = clf.score(X_test, y_test)
    print("\nScore for test set: ", score_test)
    
    matrix = confusion_matrix(y_test, predicts)
    print ("\nConfusion matrix:..................... \n",matrix)
    
    sum_colum = np.sum(matrix, axis = 0)
   # print("\nsum_column:", sum_colum)
    sum_row = np.sum(matrix, axis = 1)
    #print("\nsum_row:", sum_row)
    TP = [matrix[i,i] for i in range(0, len(matrix))]
    print("\nTP: ", TP,"\n")   
    FP = [sum_colum[i] - matrix[i,i] for i in range(0, len(matrix))]
    print("FP: ", FP,"\n")
    FN = [sum_row[i] - matrix[i,i] for i in range(0, len(matrix))]
    print("FN: ", FN,"\n")
    Presision = [TP[i] /(TP[i] + FP[i])  for i in range(0, len(matrix))]
    Recall = [TP[i] /(TP[i] + FN[i])  for i in range(0, len(matrix))]
    F1_score = [2 * Presision[i] * Recall[i] /(Presision[i] + Recall[i])  for i in range(0, len(matrix))]
    
    print("\nPrecision: ", Presision,"\n")
    print("Recall: ", Recall,"\n")
    print("F1_scrore: ", F1_score, "\n")
    
    
#     matrix_ratio = matrix/matrix.sum(1, keepdims=True)
#     print(matrix)
#     print("\n", "Confusion matrix ratio:")
#     print(matrix_ratio)
#     print("\n", "Horizontal of confusion matrix ratio:")
#     hor = [matrix_ratio[i,i] for i in range(0, len(matrix_ratio))]
#     print(hor)
 

if (method == METHOD['audio_feature']):

    ##Loading  data from files
    filehandlerInput = open('processed-data/input.obj', 'rb')
    filehandlerOutput = open('processed-data/output.obj', 'rb')
    input = pickle.load(filehandlerInput)
    output = pickle.load(filehandlerOutput)

    nan_count = 0
    for i in  input:
       # print(i)
        if(np.isnan(i).any()):
            nan_count = nan_count +1
    
    print("nan_count: ", nan_count)
    
            
    # Get quantiry of each label
    
    #Remove labels that have small quantity.
    indices = [] 
    for i in range(0, len(output)):
        if output[i] >= 3:
            indices.append(i)
    input = np.delete(input, indices, axis = 0)
    output = np.delete(output, indices)
    
    
#     up_sampling_data = []
#     for i in range(0, len(output)):
#         if output[i] == 1:
#             up_sampling_data.append(input[i])
    
#     print("this is input:", input[0:3])
    
#     input = np.append(input, up_sampling_data, axis = 0)
#     output = np.append(output, [1] * len(up_sampling_data))
#     print("finish updsamplind data")
    
    y = np.bincount(output)
    ii = np.nonzero(y)[0]
    a = list(zip(ii,y[ii]))
    print("EMOTION_ANNOTATE: ", EMOTION_ANNOTATORS)
    print("\nThe quantity of each label: ", a, "\n")

    c = list(zip(input, output))
    random.shuffle(c)
    input, output = zip(*c)
    input = np.array(input)
    output = np.array(output)

    #Training and testing
    training(input, output)





