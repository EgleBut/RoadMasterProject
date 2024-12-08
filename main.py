#Reikia biblioteku:
#pip install neurokit2
#pip install pandas
#pip install numpy
import functions as fn
import os
import neurokit2 as nk
import numpy as np
import pandas as pd
#ignore warinings. Some HRV parameters cannot be estimated since 60sec interval is too short
import warnings

#sampling rate, frequency
sr = 135
#window size in sec.
ws = 5*135

parameter = 'HRV_SDSD'
# Get a list of all files in the current directory
files = os.listdir()

# Filter the list to only include files that start with "oh1_"
data_files = [f for f in files if f.startswith("oh1_")]

dataframes = []
file_names = []
# Open just the first file and read all the rada to pandas dataframe
for i in range(len(data_files)):
  #print(data_files[i])
  df_data = fn.open_plot(data_files[i], 0)
  if fn.validate_signal(df_data['ppg0'])[0]:
    dataframes.append(df_data)
    file_names.append(data_files[i])
print(f"Number of files that were valid: {len(dataframes)}")
print(f"File names that were valid {len(file_names)}")

# Trying not to lose original values in the txt file
dataframes1 = dataframes

idx = 1
index_to_remove = []
HRV2_list = []
for i in range(len(dataframes1)):
  print(f'Working with {idx}/{len(dataframes1)} dataframe')
  warnings.filterwarnings('ignore')
  HRV2 = fn.estimateHRV(dataframes1[i]['ppg0'], sr, ws)
  if len(HRV2) != 0 or HRV2.shape[1] != 0:
    HRV2_list.append(HRV2)
    idx += 1
  else:
    index_to_remove.append(i)
    idx += 1

for i in index_to_remove:
    dataframes1.pop(i)
    file_names.pop(i)

HRV2_list_normalized = []
index_to_remove = []
for i in range(len(HRV2_list)):
  normalize_estimation = fn.normalize_all_hrv(HRV2_list[i])
  if normalize_estimation is None:
    index_to_remove.append(i)
    #HRV2_list.pop(i)
    #file_names.pop(i)
  else:
    HRV2_list_normalized.append(normalize_estimation)

for index in index_to_remove:
  HRV2_list.pop(index)
  file_names.pop(index)
  dataframes1.pop(index)

HRV2_normalized_prediction = []
for i in range(len(HRV2_list_normalized)):
  HRV2_normalized_prediction.append(fn.add_prediction_column_hrv(HRV2_list_normalized[i], dataframes1[i], parameter))

for i in range(len(HRV2_normalized_prediction)):
  print(HRV2_normalized_prediction[i].columns)
  dataframes1[i] = fn.add_prediction_column_dataframe(HRV2_normalized_prediction[i], dataframes1[i], parameter)

times_list = []
for i in range(len(dataframes1)):
  df1 = dataframes1[i]
  df1 = df1[df1['prediction']==1][['timestamps', 'prediction']]
  times_list.append(df1)

event_start_times_list = []
for i in range(len(times_list)):
  data = times_list[i]
  event_start_times = fn.count_separate_events(data)
  event_start_times_list.append(event_start_times)

video_event_times = []
for i in range(len(event_start_times_list)):
  df1 = dataframes1[i]
  event_start_times = event_start_times_list[i]
  video_event_times.append(fn.find_video_timestamps(event_start_times, df1))

video_event_times_formatted_list = []
for i in range(len(video_event_times)):
  video_event_times_i = video_event_times[i]
  video_event_times_formatted = [fn.convert_seconds_to_minutes_seconds(time) for time in video_event_times_i]
  video_event_times_formatted_list.append(video_event_times_formatted)

for i in range(len(file_names)):
  print(f'in the file {file_names[i]} these events were identified {video_event_times_formatted_list[i]} \n')