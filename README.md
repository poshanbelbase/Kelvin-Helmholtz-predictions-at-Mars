# Kelvin-Helmholtz-predictions-at-Mars
#Prediction of the KH instabilities at the Mars using machine learning tools
import numpy as np 
import pandas as pd
import pytplot
import pytplot as tplot
path=r'path to the file'
mag_vars= pytplot.sts_to_tplot(path)
times = []

for ts in time.values:
    nanos = ts.value
    secs = nanos / 1e9
    # Convert seconds to pandas Timestamp with time zone
    pd_ts = pd.Timestamp.fromtimestamp(secs, tz='US/Eastern')
    
    # Append as numpy datetime64
    times.append(pd.Timestamp.to_datetime64(pd_ts))

# Convert times to a numpy array
times = np.array(times)
times
#getting the arrays of the data
data=pytplot.get_data('diff_en_fluxes')
# data
y=data[1]
v=data[2]
y_transposed=y.T
y_transposed.shape
#Convert to pandas Series
times_series = pd.Series(pd.to_datetime(times))
time1_series = pd.Series(pd.to_datetime(time1))
time2_series = pd.Series(pd.to_datetime(time2))

# Define the date for masking (2018-01-04)
mask_date = pd.Timestamp('2017-04-03').date()

# Create the date masks
date_mask = times_series.dt.date == mask_date
date_mask1 = time1_series.dt.date == mask_date
date_mask2 = time2_series.dt.date == mask_date

# Apply the masks to the timestamp arrays
masked_times = times_series[date_mask]
masked_times1 = time1_series[date_mask1]
masked_times2 = time2_series[date_mask2]

# Apply the masks to data arrays
masked_Btot = Btot[date_mask]
masked_Bx = X[date_mask]
masked_By = Y[date_mask]
masked_Bz = Z[date_mask]
masked_y_transposed = y_transposed[:, date_mask1]
masked_ys_trans = y1_transposed[:, date_mask2]
import matplotlib.patches as mpatches
def segment_data(times, data, freq='1H'):
    df = pd.DataFrame({'times': times})
    df.set_index('times', inplace=True)
    if data.ndim == 1:
        df['data'] = data
    else:
        # Handle the case where data is a 2D array
        for i in range(data.shape[0]):
            df[f'data_{i}'] = data[i, :]
    grouped = df.resample(freq)
    return [group for _, group in grouped]

# Segment the timeseries data into one-hour intervals
segments1 = segment_data(masked_times1, masked_y_transposed, '1H')
segments2 = segment_data(masked_times2, masked_ys_trans, '1H')
segments_Btot = segment_data(masked_times, masked_Btot, '1H')

# Calculate the total number of segments (maximum of all segment lengths times 3 for each data type)
num_segments = max(len(segments1), len(segments2), len(segments_Btot)) * 3

# Create subplots
fig, axes = plt.subplots(num_segments, 1, figsize=(10, 3 * num_segments))

# Ensure axes is iterable
if num_segments == 1:
    axes = [axes]
  # Define colors for the different labels
label_colors = {
    'solar wind': 'green',
    'bow shock': 'red',
    'magnetosheath': 'blue',
    'IMB': 'yellow',
    'Induced magnetosphere': 'purple',
    'KH Vortices': 'gray'  # default color for unlabelled data
}
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in label_colors.items()]

# Plot each segment
for i in range(num_segments):
    segment_index = i // 3  # Determine the segment index for each type of plot

    # Determine which times and data to use
    if i % 3 == 0:
        # Spectrogram of SWEA
        if segment_index >= len(segments1):
            continue
        segment = segments1[segment_index]
        times_segment = segment.index.to_numpy()
        y_segment = np.log10(np.vstack(segment[[f'data_{j}' for j in range(masked_y_transposed.shape[0])]].values.T))

        # If y_segment is empty, skip the plotting
        if y_segment.size == 0:
            continue

        v_mesh = v
        times_mesh, v_mesh = np.meshgrid(times_segment, v_mesh)
        c = axes[i].pcolormesh(times_mesh, v_mesh, y_segment, shading='auto')
        ax = axes[i]

        # Create an axis for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='1%', pad=0.05)
          # If Btot_segment is empty, skip the plotting
        if Btot_segment.size == 0:
            continue

        axes[i].plot(times_segment, Btot_segment)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Btot')
        axes[i].set_title(f'B total plot {segment_index + 1}',fontsize=7)

    # Add axvspan to highlight different regions
    for start, (end, label) in manual_labels.items():
    # Check if the label region overlaps with the segment
        if (start < times_segment[-1] and end > times_segment[0]):
        # Determine the actual overlap to highlight correctly
            start_highlight = max(start, times_segment[0])
            end_highlight = min(end, times_segment[-1])
            axes[i].axvspan(start_highlight, end_highlight, color=label_colors.get(label, 'gray'), alpha=0.2)

axes[0].legend(handles=legend_patches, loc='upper left', title="Regions")

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
# Importing modules for neural network
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional, Masking

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
x_train, x_testval, y_train, y_testval = train_test_split(x, y, test_size=0.5, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, test_size=0.5, shuffle=False)
# Calculate pos_weight
total_negative = np.sum(y== 0)
total_positive = np.sum(y == 1)
POS_WEIGHT = total_negative / total_positive

# Define custom loss function
def weighted_binary_crossentropy(target, output):
    _epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    target = tf.cast(target, tf.float32)
    output = tf.cast(output, tf.float32)
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None, 1)))  # Masking layer for variable-length sequences
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
adam = Adam(learning_rate=0.001)

model.compile(loss=weighted_binary_crossentropy, optimizer=adam,  metrics=['accuracy'])
# training data, adjusting its weights. Usually more epochs is better!
model.fit(x_train, np.asarray(y_train), epochs=10, batch_size=300,validation_data = (x_val, y_val))

    
