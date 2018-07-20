
# coding: utf-8

# In[1]:


import pickle
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sys
sys.path.insert(0, "../")
from acousticFeatures import getAllFeatures
import parselmouth 
import numpy as np
from pydub import AudioSegment
from IPython.display import Audio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
from os.path import isfile


class Input:
    def __init__(self, code=None, excited=None, valance=None, arouse=None):
        self.__code = code
        self.__excited = excited
        self.__valance = valance
        self.__arouse = arouse
    
    def get_code(self):
        return self.__code
    
    def get_excited(self):
        return self.__excited
    
    def get_valance(self):
        return self.__valance
    
    def get_arouse(self):
        return self.__arouse
    
class Output:
    def __init__(self, dataFrame=None):
        self.__dataFrame = dataFrame
          
    def get_data_frame(self):
        return self.__dataFrame
        
        
    


# In[2]:


#Path input and labels.
AUDIO_FILES_PATH = "auditary_emotion_recognition/data_interview/Audio/Audio"
AUDIO_LABEL_PATH = "auditary_emotion_recognition/data_interview/Labels/turker_scores_full_interview.csv"

try:
    #Load max, min, mean, std of acoustic features.
    max_acoustic_features = pickle.load(open('processed-data/max_acoustic_features.obj', 'rb'))
    min_acoustic_features = pickle.load(open('processed-data/min_acoustic_features.obj', 'rb'))
    mean_acoustic_features = pickle.load(open('processed-data/mean_acoustic_features.obj', 'rb'))
    std_acoustic_features = pickle.load(open('processed-data/std_acoustic_features.obj', 'rb'))

    #Load model
    model_classification_excited = pickle.load(open("model/model_classification_excited.sav", 'rb'))
    model_regression_valance = pickle.load(open("model/model_regression_valance.sav", 'rb'))
    model_regression_arouse = pickle.load(open("model/model_regression_arouse.sav", 'rb'))
except:
    raise ("Getting error when loading statistic of features or trained models!!!")


# In[5]:


def create_input(code=None, frame_length=4000, frame_overlap=2000):
    file_path = AUDIO_FILES_PATH + "/" + code.upper() +".wav"
    if not isfile(file_path):
        print("File path is invalid")
    sound = AudioSegment.from_file(file_path)
    
    input = []
    s = 0
    e = frame_length
    while (True):
        chunk = sound[s:e]
        left, right = chunk.split_to_mono()
        sound_frame = parselmouth.Sound(left.get_array_of_samples(), sound.frame_rate)

        # Todo
        acoustic_features = np.array(getAllFeatures(sound_frame))
        if(acoustic_features is not None and len(acoustic_features) > 10):
            normalized_acoustic_features = (acoustic_features - min_acoustic_features) / (max_acoustic_features - min_acoustic_features)
            input.append(normalized_acoustic_features)
        
        # Update start frame and end frame
        s = e - frame_overlap
        e = s + frame_length
        
        if (e > sound.duration_seconds * 1000):
            break
    
    # Fill Nan values by mean values of columns
    input = np.array(input)
    col_mean = np.nanmean(input, axis=0)
    inds = np.where(np.isnan(input))
    input[inds] = np.take(col_mean, inds[1])

    # Predict by frames.
    excited = np.array(model_classification_excited.predict(np.array(input)))
    valance = np.array(model_regression_valance.predict(np.array(input)))
    arouse = np.array(model_regression_arouse.predict(np.array(input)))
    
    return  Input(code, excited, valance, arouse)

def create_output(data_frame=None):
    return Output(data_frame)


# In[6]:


# create output
df = pd.read_csv(AUDIO_LABEL_PATH)
output = create_output(df)

#create input

input = []

for key in df.Participant.unique():
    input.append(create_input(key))
    print("Finish {}".format(key))

pickle.dump(input, open('processed-data/input-interview-frameLengh4-overlap2.obj', 'wb'))
pickle.dump(output, open('processed-data/output-interview.obj', 'wb'))
    


# In[44]:





# In[ ]:




