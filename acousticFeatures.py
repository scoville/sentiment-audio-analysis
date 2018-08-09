import pydub 
import numpy as np
import sys
import parselmouth 

# Get Jittle
def calculateJitter(data):
    """Data is list of time of peaks"""
    data = np.array(data)
    data = data[data != 0]
    n = len(data)
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i > 0:
            sum1 += abs(data[i-1] - data[i])
        sum2 += data[i]
    sum1 /= float(n - 1)
    sum2 /= float(n)
    return 100 * (sum1 / sum2)


#Get Shimmer
def calculateShimmer(data):
    data = np.array(data)
    data = data[data != 0]
    n = len(data)
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i > 0:
            sum1 += abs(data[i-1] - data[i])
        sum2 += data[i]
    sum1 /= float(n - 1)
    sum2 /= float(n)
    return 100 * (sum1 / sum2)

def getStatistic(numpy_arr):
    numpy_arr = np.array(numpy_arr)
    numpy_arr = numpy_arr[numpy_arr != 0]
    max_v = np.max(numpy_arr)
    min_v = np.min(numpy_arr)
    range_v = np.max(numpy_arr) - np.min(numpy_arr)
    mean_v = np.mean(numpy_arr)
    median_v = np.median(numpy_arr)
    per25_v = np.percentile(numpy_arr, 25)
    per75_v= np.percentile(numpy_arr, 75)
    std_v = np.std(numpy_arr)
    return np.array([max_v, min_v, range_v, mean_v, median_v, per25_v, per75_v, std_v])


def estimate_voiced_unvoiced_and_breaks(file, THRESHOLD_UNVOICED = 0.1 ):
    time_step = 0.01
    snd = parselmouth.Sound(file)
    pitch = snd.to_pitch(voicing_threshold=0.45 ,silence_threshold = 0.03, time_step = time_step)
    amplitude_arr = pitch.selected_array['strength']
    num_breaks = []
    num_unvoiced = []
    num_voiced = []
    temp = []
    i = 0
    while(i < len(amplitude_arr)-1):
        i += 1
        if (len(temp) == 0):
            temp.append(amplitude_arr[i])
            
                
        if (i < len(amplitude_arr) and (temp[0] == 0.0 and amplitude_arr[i] == 0.0) or (temp[0] !=0 and amplitude_arr[i]!=0)):
            temp.append(amplitude_arr[i])

        else:
            if (temp[0] != 0):
                num_voiced.append(len(temp) * time_step)
            else:  
                if (len(temp) >= THRESHOLD_UNVOICED / time_step):
                    num_unvoiced.append(len(temp) * time_step)
                else:
                    num_breaks.append(len(temp)  * time_step)
            
            temp = []
            
            #continue
        
    
            
    return num_voiced, num_unvoiced, num_breaks


feature_name= np.array(['energy', 
               'f0', 'intensity', 'f1', 'f2', 'f3','f1-bw','f2-bw','f3-bw' ,
               'f2-f1', 'f3-f1', 
               'jitter', 'shimmer', 'duration',
              'unvoiced_percent', 'breaks_degree', 'max_dur_pause', 'average_dur_pause'])


def getAllFeatures(file):
    try:
        features = []

        #Get peaks and that of frames and times.
        snd = parselmouth.Sound(file)
        pitch = snd.to_pitch(time_step = 0.01)
        formants = snd.to_formant_burg()
        num_frames = pitch.get_number_of_frames()
        frames = [pitch.get_frame(i) for i in range(1, num_frames+1)]
        times = [pitch.get_time_from_frame_number(i) for i in range(1, num_frames+1)]

        #Get energy
#         energy = snd.get_energy()
#         if('energy' in feature_name):
#             features.append(energy)

        #Get F0 statitic
        f0_arr = pitch.selected_array['frequency']
        f0_stat = getStatistic(f0_arr)
        if('f0' in feature_name):
            features = np.append(features, f0_stat)

        #Get intensity statistic
        intensity_arr = [frame.intensity for frame in frames]
        intensity_stat = getStatistic(intensity_arr)
        if('intensity' in feature_name):
            features = np.append(features, intensity_stat)


        #Get formant values and format bandwidth statistic
        f1_arr = [formants.get_value_at_time(1, time) for time in times]
        f1_bandwidth_arr = [formants.get_bandwidth_at_time(1, time) for time in times]
        f1_stat = getStatistic(f1_arr)
        f1_bandwidth_stat = getStatistic(f1_bandwidth_arr)
        if('f1' in feature_name):
            features = np.append(features, f1_stat)
        if('f1-bw' in feature_name):
            features = np.append(features, f1_bandwidth_stat)

        f2_arr = [formants.get_value_at_time(2, time) for time in times]
        f2_bandwidth_arr = [formants.get_bandwidth_at_time(2, time) for time in times]
        f2_stat = getStatistic(f2_arr)
        f2_bandwidth_stat = getStatistic(f2_bandwidth_arr)
        if('f2' in feature_name):
            features = np.append(features, f2_stat)
        if('f2-bw' in feature_name):
            features = np.append(features, f2_bandwidth_stat)

        f3_arr = [formants.get_value_at_time(3, time) for time in times]
        f3_bandwidth_arr = [formants.get_bandwidth_at_time(3, time) for time in times]
        f3_stat = getStatistic(f3_arr)
        f3_bandwidth_stat = getStatistic(f3_bandwidth_arr)
        if('f3' in feature_name):
            features = np.append(features, f3_stat)
        if('f3-bw' in feature_name):
            features = np.append(features, f3_bandwidth_stat)
           
        #f2/f1, f3/f1 statistic
        f2_over_f1_arr = np.array(f2_arr) / (np.array(f1_arr) + 1)
        f3_over_f1_arr = np.array(f3_arr) / (np.array(f1_arr) + 1)
        f2_over_f1_stat = getStatistic(f2_over_f1_arr)
        f3_over_f1_stat = getStatistic(f3_over_f1_arr)
        if('f2-f1' in feature_name):
            features = np.append(features, f2_over_f1_stat)
        if('f3-f1' in feature_name):
            features = np.append(features, f3_over_f1_stat)
           

        #Jitter
        f0_arr = np.array(f0_arr)
        f0_arr = f0_arr[f0_arr !=  0]
        jitter = calculateJitter(1000/f0_arr)
        if('jitter' in feature_name):
            features = np.append(features, jitter)

        #Shimmer
        amplitude_arr = pitch.selected_array['strength']
        shimmer = calculateShimmer(amplitude_arr)
        if('shimmer' in feature_name):
            features = np.append(features, shimmer)

        #Duration
        duration = snd.duration
        if('duration' in feature_name):
            features = np.append(features, duration)
            
            
        # Get voiced, unvoiced, break periods
        voices, unvoices, breaks = estimate_voiced_unvoiced_and_breaks(file, THRESHOLD_UNVOICED = 0.5) #if amplitude is == 0 in at least 0.5s --> unvoiced
        unvoiced_percent = (np.sum(unvoices)  + np.sum(breaks))/ duration + 0.01
        breaks_degree = np.sum(breaks) / (np.sum(voices) + np.sum(breaks) + 0.01)
       
        if(len(breaks) != 0):
            max_dur_pause = np.max(breaks)
            average_dur_pause  = np.average(breaks)
        else:
            max_dur_pause = 0
            average_dur_pause = 0
            breaks_degree = 0
            

#         if('unvoiced_percent' in feature_name):
#             features = np.append(features, unvoiced_percent)  
#         if('breaks_degree' in feature_name):
#             features = np.append(features, breaks_degree)
#         if('max_dur_pause' in feature_name):
#             features = np.append(features, max_dur_pause)
#         if('average_dur_pause' in feature_name):
#             features = np.append(features, average_dur_pause)
                
    except :
        print("May be file is so short: ", file)
        return np.array([None])
   
    return features




