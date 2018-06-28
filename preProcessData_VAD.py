
# coding: utf-8

# In[ ]:


import os
import re
import glob 
import matplotlib.pyplot as plt
import numpy as np
from os.path import basename
#import audiosegment
from multiprocessing import Pool,freeze_support

import sys
import numpy

import librosa
import pickle
import acousticFeatures


#Constant
EMOTION_ANNOTATORS = {'anger': 0, 'happiness' : 1, 'sadness' : 2, 'neutral' : 3, 'frustration' : 4, 'excited': 5,
           'fear' : 6,'surprise' : 7,'disgust' : 8, 'other' : 9}

EMOTION = {'ang': 0, 'hap' : 1, 'sad' : 2, 'neu' : 3, 'fru' : 4, 'exc': 5,
           'fea' : 6,'sur' : 7,'dis' : 8, 'oth' : 9, 'xxx':10}

METHOD = {'audio_feature':0, 'LSTM':1}

#Method for classification
method = METHOD['audio_feature']


isRawDataProcessed = False

#Development mode. Only run with small data.
dev = False

onlyAcoustic = True


#Define class
class Input:
    ##spectral, prosody, erergy are dict type
    def __init__(self, spectral=None, prosody=None, energy=None, spectrogram=None, acoustic_features=None):
        self.spectral = spectral
        self.prosody = prosody
        self.energy = energy
        self.spectrogram = spectrogram
        self.onlyAcoustic = onlyAcoustic
        self.acoustic_features = acoustic_features
        
    def print(self):
        print("spectral  features: ", spectral)
        print("prosody features: ", prosody)
        print("energy: ", energy)
        print("spectrogram: ", spectrogram)
        
    def input2Vec(self, onlySpectrogram, onlyAcoustic):
        
        if (onlySpectrogram ==  False):
            features = []
            if (onlyAcoustic == False):
                s = list(self.spectral.values())
                p = list(self.prosody.values())
                e = list(self.energy.values())
                [features.extend(x) for x in [s, p, e]]
            else:
                features = self.acoustic_features
               # print("fea:", features)
            return features
        else :
            return self.spectrogram
    
class Output:
    def __init__(self, duration, code, category_origin, category_evaluation, attribute, ):
        self.duration = duration
        self.code = code
        self.category_origin = category_origin
        self.category_evaluation = category_evaluation
        self.attribute = attribute
        
    def isConsitent(self):
        counts = np.unique(self.category_evaluation , return_counts=True)[1]
        consitent_degree = np.max(counts)/np.sum(counts)
        print(consitent_degree)
        if(consitent_degree >= 0.7):
            return True
        return False

    def output2Vec(self):
        return self.attribute
    
    
    
#Functions for get features from audio file
def amp2Db(samples):
    dbs = []
    for  x in samples:
        if x < 0:
            v = - dspUtil.rmsToDb(np.abs(x))
        elif x == 0:
            v = 0
        else :
            v = dspUtil.rmsToDb(np.abs(x))
        dbs.append(v)
    return dbs

def getF0Features(file):
    features = {}
    sound = audiosegment.from_file(file)
    voiced = sound.filter_silence(duration_s=0.2)
    frame_rate = sound.frame_rate
    frames = sound.dice(0.032)

    f0s = []
    for f in frames:
        f0 = dspUtil.calculateF0once(amp2Db(f.get_array_of_samples()), frame_rate)
        if(f0 != 0):
            f0s.append(f0)
    
    features['f0_min'] = np.min(f0s)
    features['f0_max'] = np.max(f0s)
    features['f0_range'] = np.max(f0s) - np.min(f0s)
    features['f0_mean'] = np.mean(f0s)
    features['f0_median'] = np.median(f0s)
    features['f0_25th'] = np.percentile(f0s, 25)
    features['f0_75th'] = np.percentile(f0s, 75)
    features['f0_std'] = np.std(f0s)
    
  
    return features

def getEnergyFeatures(file):
    features = {}
    sound = audiosegment.from_file(file)
    voiced = sound.filter_silence(duration_s=0.2)
    samples = voiced.get_array_of_samples()
    frame_rate = sound.frame_rate
    frames = sound.dice(0.032)
    
    e = []
    for f in frames:
        e.append(np.abs(f.max_dBFS))
    
    
    features['energy_min'] = np.min(e)
    features['energy_max'] = np.max(e)
    features['energy_range'] = np.max(e) - np.min(e)
    features['energy_mean'] = np.mean(e)
    features['energy_median'] = np.median(e)
    features['energy_25th'] = np.percentile(e, 25)
    features['energy_75th'] = np.percentile(e, 75)
    features['energy_std'] = np.std(e)   

    return features
    
def audio2Features(file):
    spectral = {}
    prosody = {}
    energy = {}
    try:
        if (onlyAcoustic == False):
            prosody = getF0Features(file)
            energy = getEnergyFeatures(file)
            y, sr = librosa.load(file)
            spectrogram = librosa.stft(y)
            spectrogram = np.abs(spectrogram)
            #To be continued....
            return Input(spectral, prosody, energy, spectrogram)
        else :
            acoustic_features = acousticFeatures.getAllFeatures(file)
            return Input(acoustic_features = acoustic_features)
    except Exception as e:
        print(e)
        
        
#Function for getting input vector and corresponding output      
def parallel_task(d0, d1):
    print("task...")
    # Each input diectory contains many file
    # This fucntion will walk through all valid 'wav'files in this directory and get features like engergy, frequency...
    def parseInput(dir):
        dicts = {} 
        for f in os.listdir(dir):
            if not f.startswith(".") and os.path.splitext(f)[1] == ".wav":
                dicts[os.path.splitext(f)[0]] = audio2Features(dir + "/" + f)


        return dicts
    
    # Each output file contains label of many diffrent 'wav' file.
    # This function will parse content of text file using 'regrex'. Then turn it into label
    def parseOutput(file):
        dict_namefile_output = {}
        # Open file to get all contents excepts the first line.
        f = open(file, 'r')
        content = ""
        index = 0
        for line in f:
            index = index + 1
            if index == 1:
                continue
            content  = content + line

        # Find all matched patterns in the content
        ps = re.findall(r'\[.*?\)\n\n', content, re.DOTALL)

        # Parse each matched pattern into  'Output' object
        try:
            for p in ps:
                ls = p.split("\n")
                ls = list(filter(lambda x: len(x) > 0 ,ls))

                # Split elements of the first line which looks like : 
                # [147.0300 - 151.7101]	Ses01F_impro02_M012	neu	[2.5000, 2.0000, 2.0000]
                ele_line0 = re.search(r'(\[.*?\])(\s)(.*?)(\s)(.*?)(\s)(\[.*?\])', ls[0]).groups()

                # Split time components which looks like:
                # [147.0300 - 151.7101]
                time_dur = ele_line0[0]
                ele_time_dur = re.findall(r"[-+]?\d*\.\d+|\d+", time_dur)
                ele_time_dur = [float(x) for x in ele_time_dur]

                # Get code and category_origin which looks like:
                # Code: Ses01F_impro02_M012
                # Category_origin: neu
                code = ele_line0[2]
                category_origin = ele_line0[4]

                # Split attribute components which looks like:
                # [2.5000, 2.0000, 2.0000]
                attribute = ele_line0[6]
                ele_attribute = re.findall(r"[-+]?\d*\.\d+|\d+", attribute)
                ele_attribute = [float(x) for x in ele_attribute]

                # Get categorial_evaluation:
                lines_categorical = list(filter(lambda x : x[0] == 'C', ls))
                rex = re.compile(r'C.*?:(\s)(.*?)(\s)\(.*?\)')

                category_evaluation = []
                for l in lines_categorical:
                    elements = rex.search(l).groups()
                    cat = elements[1]
                    cat = cat.split(";")
                    cat = map(lambda x: x.lstrip(), cat)
                    cat = list(filter(lambda x: len(x)>0, cat))
                    category_evaluation.extend(cat)


                # Make list distinct
                category_evaluation = np.array(category_evaluation)
                #category_evaluation = list(set(category_evaluation))
                
                

                # Make dict {name_file : parsed_output}
                dict_namefile_output[code] = Output(ele_time_dur, code, category_origin, category_evaluation, ele_attribute)
            return dict_namefile_output
        except Exception as e:
            print(e)


    ### Parse input and output files and get input and output as vector
    dicts_in = parseInput(d0)
    dicts_out = parseOutput(d1)
    in_out = []
    
    keys = list(dicts_in.keys())
    for key in keys:
           
        value = dicts_in[key].input2Vec(onlySpectrogram=False, onlyAcoustic=True)
        if(value.all() != None ):
            in_out.append((value, dicts_out[key].output2Vec()))
    return in_out
    
    
def createInput_Output():
    ### Get directories of input and output
    DATA_DIR = "auditary_emotion_recognition/IEMOCAP_full_release"
    NUM_SESSION = 5
    input_output = []
    for i in range (1, NUM_SESSION + 1):
        name_session = "Session" + str(i)
        root_dir_of_wav = DATA_DIR + "/" + name_session + "/sentences" + "/wav"
        root_dir_of_labels = DATA_DIR + "/" + name_session + "/dialog" + "/EmoEvaluation"

        for x in os.walk(root_dir_of_wav):
            if(x[0] == root_dir_of_wav):
                dirs_of_wav = x[1]
                index = -1
            else:
                index = index + 1
                input_output.append((x[0], root_dir_of_labels + "/" + dirs_of_wav[index] + ".txt"))
                
    
    ds = input_output
    in_out = []
    input = []
    out = []
    
    # Multi processing
    with Pool(processes=8) as pool:
         in_out = pool.starmap(parallel_task, ds)
   
    r = []
    for e in in_out:
        r = r + e
    
    input = [x[0] for x in r]
    out = [x[1] for x in r]
    print("Finished creating input output into txt file")
    print("len input and output:", len(input), ", ", len(out))
    return (input, out)
 


#If have not processed data yet then process, otherwise loading data from file.
if __name__ == '__main__':
   # multiprocessing.set_start_method('folk')
    print("hello")
    freeze_support()
    if isRawDataProcessed == False:

        ##Get input, normalize input, get output
        input, output = createInput_Output()
        output = np.array(output)

        if(method == METHOD['audio_feature']):
            input = np.array(input)
           # input = input / input.max(axis=0)
            filehandlerInput = open('processed-data/input_VAD.obj', 'wb')
            filehandlerOutput = open('processed-data/output_VAD.obj', 'wb')


        pickle.dump(input, filehandlerInput)
        pickle.dump(output, filehandlerOutput)
        print("Finish write processed data (input, output) to file!!!")

