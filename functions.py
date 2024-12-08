#To check if signal is not too short for the analysis. If it is too short, remove it

import pandas as pd
from datetime import datetime
import ast
import numpy as np
import neurokit2 as nk

#sampling rate, frequency
sr = 135
#window size in sec.
ws = 5*135

#parameter = 'HRV_SDSD'

my_columns = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2',
       'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD',
       'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
       'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50',
       'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_ULF',
       'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_TP', 'HRV_LFHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2',
       'HRV_S', 'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP',
       'HRV_IALS', 'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI',
       'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d',
       'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd',
       'HRV_SDNNa', 'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width',
       'HRV_MFDFA_alpha1_Peak', 'HRV_MFDFA_alpha1_Mean',
       'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Delta',
       'HRV_MFDFA_alpha1_Asymmetry', 'HRV_MFDFA_alpha1_Fluctuation',
       'HRV_MFDFA_alpha1_Increment', 'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn',
       'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD',
       'HRV_HFD', 'HRV_KFD', 'HRV_LZC']

def validate_signal(signal):
    if len(signal) < 60 * 135 * 5:  # 10 minutes * 60 seconds * 135 Hz
        print(len(signal))
        return False, "Signal is too short"
    try:
        # Convert all elements to float
        signal = [float(value) for value in signal]
    except ValueError as e:
        return False, f"Signal contains non-numeric values: {e}"
    # Check how many values are higher than 1e8
    count_higher_than_1e7 = sum(1 for value in signal if value > 1e7)
    if count_higher_than_1e7 > 200:
        return False, "Signal is not valid"
    return True, "Signal is valid"
#open data file, read data and add to pandas dataframe.
#if flag = 1, plot data, othervise, just return the data
def open_plot(file_name, flag):
  ppg0 = []
  ppg1 = []
  ppg2 = []
  ts = []
  ambient = []
  timestamps = []
  with open(file_name, 'r') as csvf:
    print(csvf.readline())
    for line in csvf:
      try:
        data_string = line[20:].strip()
        if data_string:  # Ensure there's something to evaluate
          data = ast.literal_eval(line[20:])
          timestamps.append(datetime.strptime(line[:19], '%Y_%m_%d-%H-%M-%S')) #2024_07_26-10-43-39
          ppg0.append(data["ppg0"])
          ppg1.append(data["ppg1"])
          ppg2.append(data["ppg2"])
          ambient.append(data["ambient"])
          ts.append(data["ts"])
      except SyntaxError:
          print(f"Skipping malformed data line: {line}")
      except Exception as e:
          print(f"An error occurred: {e}")

    dictionary = {'ppg0': ppg0, 'ppg1': ppg1, 'ppg2': ppg2, 'ambient': ambient, 'ts': ts, 'timestamps': timestamps}
    df = pd.DataFrame(dictionary)
  check = validate_signal(ppg0)
  print(check[1])
  return df

def estimateHrvMinutes(epochs):
  hrv_param = []
  stop = 1
  i = 0
  nan_row = pd.DataFrame([[np.nan]*len(my_columns)], columns=my_columns) #if something goes wrong
  for epoch in epochs:
    if stop >= len(epochs):
      break
    stop += 1
    i += 1
    try:
      if not epochs[epoch].empty:
        hrv = nk.hrv(epochs[epoch]['PPG_Peaks'], sampling_rate=sr)
        hrv_param.append(hrv)
      else:
        hrv_param.append(nan_row)
        print(f"Skipping empty dataset at index {i}")

      #hrv = nk.hrv(epochs[epoch]['PPG_Peaks'], sampling_rate=sr) #non linear parameters have high number of inf, nan values
      #hrv = nk.hrv_time(epochs[epoch]['PPG_Peaks'], sampling_rate=sr)
    except ZeroDivisionError:
      #print('ZeroDivisionError')
      hrv_param.append(nan_row)
      pass
    except ValueError:
      #print('ValueError')
      hrv_param.append(nan_row)
      pass
    except IndexError as ie:
      #print(f"IndexError")
      hrv_param.append(nan_row)
      pass#print(f"IndexError at index {i}: {ie}")
    except Exception as e:
      hrv_param.append(nan_row)
      pass#print(f"General error at index {i}: {e}")
  return hrv_param

def addToOneFrame(hrv_param):
  line = 1
  hrv_full = hrv_param[0]
  #print(type(hrv_full))
  while line < len(hrv_param):
    hrv_full = pd.concat([hrv_full, hrv_param[line]], ignore_index = True)#_append
    line += 1
  return hrv_full

def estimateHRV(signal, sr, window):
  #If sliding window is equal to 1 min, than intervals in epochs do not overlap and whole signal can be analysed
  #If sliding window is less than 1 min than signal values overlap in every epoch and due to zero values in the last epohs errors occur
  sig_end = 0
  if window < 60*sr:
    sig_end = 60*sr-window


  p1c = nk.ppg_clean(signal, sampling_rate=sr,  method='elgendi')
  p1p, p1pinfo = nk.ppg_peaks(p1c, sampling_rate=sr, method='elgendi', correct_artifacts=True)
  artifacts, p1pfl = nk.signal_fixpeaks(p1p, sampling_rate=sr, method="neurokit", relative_interval_max=1.5)

  p1pfv = np.zeros(len(p1p))
  p1pfv[p1pfl] = 1
  p1pf = pd.DataFrame({'PPG_Peaks': p1pfv})
  p1rf = nk.signal_rate( p1pf, sampling_rate=sr, desired_length=len(p1p))
  sig = pd.DataFrame({'PPG_Raw': signal, 'PPG_Clean':p1c, 'PPG_Rate':p1rf, 'PPG_Peaks':p1p.values[:,0]})
  events = np.arange(0, len(sig)-sig_end, window).astype(int) #markers for every minute split.
  epochs = nk.epochs_create(sig, events, sampling_rate=sr, epochs_start=0, epochs_end=60 )
  hrv = estimateHrvMinutes(epochs)
  if len(hrv) == 0:
    return []
  else:
    hrv_full = addToOneFrame(hrv)
    hrv_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    hrv_full.dropna(how='all', axis=1, inplace=True)
    return hrv_full
# function that normalize HRV2 data

def normalize_all_hrv(hrv_data):
  normalized_data = hrv_data
  for parameter in hrv_data.columns:
    if not isinstance(hrv_data[parameter], pd.Series):
        print("Error: Input must be a pandas Series.")
        return None

    if hrv_data[parameter].empty:
        print("Error: Input Series is empty.")
        return None

    # Handle potential NaN values
    hrv_data[parameter] = hrv_data[parameter].dropna()
    if hrv_data.empty:
        print("Error: Input Series contains only NaN values.")
        return None

    #min-max normalization
    min_val = hrv_data[parameter].min()
    max_val = hrv_data[parameter].max()

    if min_val == max_val:
        print("Warning: Minimum and maximum values are equal. Cannot normalize.")
        return None

    normalized_data[parameter] = (hrv_data[parameter] - min_val) / (max_val - min_val)
  return normalized_data

# add 'prediction' column to the normalised dataframe with values 1 if the stress is identified and 0 if stress is absent. The stress is identified if the outlier appeared. However, the stressful state is not that short, it is only a peak, so add ones starting from outlier to the average value in both sides

# Assuming 'normalised' is your normalized HRV_SDSD data and 'df1' is your original dataframe
# and 'ws' is the window size used in the calculations (from the provided code).

def add_prediction_column_hrv(HRV2_normalized, original_df, parameter):
    """Adds a 'prediction' column to the dataframe based on identified outliers."""
    hrv_analysed = HRV2_normalized[parameter]
    # Calculate average
    average_value = np.mean(hrv_analysed)
    std_value = np.std(hrv_analysed)

    # Find outliers using Z-score
    z_scores = (hrv_analysed - np.mean(hrv_analysed)) / np.std(hrv_analysed)
    outliers_indices = np.where(np.abs(z_scores) > 3)[0]

    # Initialize the prediction column with zeros
    predictions = np.zeros(len(hrv_analysed))

    for outlier_index in outliers_indices:
        # Find the start and end points for the stress period
        start_index = outlier_index
        while start_index > 0 and hrv_analysed[start_index-1] > average_value + std_value:
          start_index -= 1
        end_index = outlier_index
        while end_index < len(hrv_analysed)-1 and hrv_analysed[end_index+1] > average_value + std_value:
          end_index += 1
        # Set predictions to 1 for the identified stress period
        predictions[start_index:end_index+1] = 1

    # Create a new DataFrame with the prediction column.
    # The length of the prediction column may be shorter than the original data if windows were skipped earlier.
    # Find the correct alignment by the length of the input data.
    x = np.arange(60 * sr, len(original_df["ppg0"]), ws)
    if len(predictions) > len(x):
        predictions = predictions[:len(x)]
    else:
        x = x[:len(predictions)]

    # Create a DataFrame from the aligned predictions
    HRV2_normalized['prediction'] = predictions
    #prediction_df = pd.DataFrame({'HRV_param': , 'prediction': predictions})

    return HRV2_normalized

def add_prediction_column_dataframe(HRV2_normalized, dataframe, parameter):
    """Adds a 'prediction' column to the dataframe based on identified outliers."""
    #hrv_analysed = HRV2_normalized[parameter]
    predictions = []
    current_prediction_index = 0
    for i in range(0, len(dataframe), ws):
      if current_prediction_index < len(HRV2_normalized):
        current_prediction = HRV2_normalized['prediction'].iloc[current_prediction_index]
        predictions.extend([current_prediction] * ws)  # Repeat prediction for ws samples
        current_prediction_index += 1
      else:
        # Handle cases where there are fewer predictions than needed for dataframe
        # For instance, extend with the last prediction or a default value
        predictions.extend([predictions[-1]] * ws) # Extend with last prediction

    # Pad predictions if dataframe is longer than expected
    predictions = predictions[len(predictions) - len(dataframe):]
    dataframe['prediction'] = predictions
    return dataframe

# Assuming 'get_time' DataFrame is already created as in your provided code.
# Sample data (replace with your actual 'get_time' DataFrame)

def count_separate_events(data):
    # Convert timestamps to datetime objects if they are not already
    if not pd.api.types.is_datetime64_any_dtype(data['timestamps']):
      data['timestamps'] = pd.to_datetime(data['timestamps'])

    unique_timestamps = data['timestamps'].unique()
    events = []
    event_start = None
    previous_timestamp = None
    #current_timestamp = unique_timestamps['timestamps'].iloc[0]

    for i in range(len(unique_timestamps)):
        current_timestamp = unique_timestamps[i]#unique_timestamps['timestamps'].iloc[i]

        if event_start is None:
            event_start = current_timestamp
            events.append(event_start)
            previous_timestamp = event_start
        else:
            time_diff = current_timestamp - previous_timestamp
            if time_diff.total_seconds() > 3:
                event_start = current_timestamp
                events.append(event_start)
                previous_timestamp = event_start
            else:
                previous_timestamp = current_timestamp

    return events

# My videos start with timestamp 0 and the signal starts with the first timestamp in df1. I already estimated when events appeared in event_start_times, but what would be the time in the video

# Assuming event_start_times is a list of datetime objects representing the start times of events in the PPG signal.
# Also assuming df1['timestamps'] represents the timestamps of the PPG signal, aligned with the video timestamps.
# The video starts at timestamp 0, and df1['timestamps'] starts at the first timestamp of the PPG signal.

# Function to find video timestamp corresponding to event timestamps
def find_video_timestamps(event_start_times, df1):
  video_timestamps = []
  for event_time in event_start_times:
    # Find the closest timestamp in df1 to the event time
    closest_index = (df1['timestamps'] - event_time).abs().idxmin()

    #Get the timestamp from df1 corresponding to the closest index
    closest_df1_timestamp = df1['timestamps'].iloc[closest_index]

    # Calculate the difference between the event timestamp and the closest timestamp in df1
    time_diff = event_time - closest_df1_timestamp
    #print(f"Time Difference: {time_diff}")

    #Retrieve the video timestamp at that index.
    #Assuming df1 has a column that corresponds to the video timestamp (e.g., 'video_timestamp'). If not, adjust this line accordingly.

    #If df1 doesn't have 'video_timestamp' column, we'll make the assumption the first timestamp in df1 is time 0
    #So, the video timestamp is equal to the df1 timestamp - the first df1 timestamp.
    video_time = closest_df1_timestamp - df1['timestamps'].iloc[0]

    video_timestamps.append(video_time.total_seconds())  # Convert to seconds

  return video_timestamps

def convert_seconds_to_minutes_seconds(seconds):
  minutes = int(seconds // 60)
  seconds = int(seconds % 60)
  return f"{minutes}:{seconds:02d}"

