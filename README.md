# Sentiment Audio Analysis
Evaluating the exciting intensity on both short audio segment(says 5s) and long audio(say 5 minutes) in an interview record. I divide my work into 2 steps:
- Building some models in an data source(IEMOCAP database  https://sail.usc.edu/iemocap/) that includes short audio segments and analyzing the model performance in this data source.
- Using models  trained in IEMOCAP to evaluate the exciting intensity in long audio belonging to interview data source (data source showed in https://arxiv.org/abs/1504.03425) 

**Note**: All data sources presented above is available on MLPC : “/data/auditary_emotion_recognition”

# How to run 2 steps above:

## 1. Create data:
You need to create  the folder name “auditary_emotion_recognition” on the root of the project and data in it. If you run all MLPC, you should use “soft link” to link the data in “/data/auditary_emotion_recognition” into your project without copying in order to save memory.

## 2. Flow to run the first step:
- Run the file “preprocess_data.ipynb”. This module preprocess raw data of IEMOCAP and create input, output which are stored in “processed-data” directory and used for training models.  **Note**: If the ‘input.obj’, ‘ouput.obj’ have existed in “processed-data” directory already, then do not need to run this file.
- Run the file “emotion_analysis.ipynb” to train models and see some analysis.

## 3. Flow to run the second step:
- Run the file “preprocess_data_interview_data.ipynb”. This module will used models trained in IEMOCAP (in the first step) to apply to every small segment of each interview record in interview data source. **Note**: If the ‘input-interview-frameLengh4-overlap2.obj’ have existed in “processed-data” directory already, then do not need to run this file 
- Run the file “analyzing_relationship_frame_and_interview_level.ipynb” to see the relationship between predicted values in frame level and true values in interview level

# Personal Idea:
If we have our own interview data set (Japanese interview records), we should try to train model directly from interview data set like the way  https://arxiv.org/abs/1504.03425 did,  and compare with the method i did which inference the values in interview level based on the predicted values in frame level. I believe that the former method would be more accurate.





