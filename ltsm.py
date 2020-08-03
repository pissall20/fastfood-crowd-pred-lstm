import numpy as np
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.models import model_from_json
# using keras backend = theano (change file in folder C:\Users\User\.keras)
from keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator
from matplotlib import pyplot
from pandas import read_csv, to_datetime, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# load dataset
df = read_csv('fastfood.csv', header=0)

# Save the last time stamp before deleting that column
# df["timestamp"] = to_datetime(df["timestamp"], format="%d/%m/%y %H:%M")

df["timestamp"] = to_datetime(df["timestamp"])
print(df.head())
base_timestamp = df["timestamp"].max()
exit()
del df['timestamp']

values = df.values
# integer encode direction
encoder = LabelEncoder()
values[:, 1] = encoder.fit_transform(values[:, 1])

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# Target vector is a set of values 480 timesteps (1*24*20) out in the future.
# 1*24*20 = 480
X = scaled[:][:-480]  # All of the above columns and remove readings from last 24 hours
y = scaled[:, 0][480:]  # Appliances or the target/label column and remove readings from first 24 hours

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.40, random_state=42, shuffle=False)

# Create overlapping windows of lagged values for training and testing datasets
# 5*24*20 = 2400 -> number of 3 minute interval in 5 days
timesteps = 2400
train_generator = TimeseriesGenerator(trainX, trainY, length=timesteps, sampling_rate=1, batch_size=timesteps)
test_generator = TimeseriesGenerator(testX, testY, length=timesteps, sampling_rate=1, batch_size=timesteps)

train_X, train_y = train_generator[0]
test_X, test_y = test_generator[0]

train_samples = train_X.shape[0] * len(train_generator)
test_samples = test_X.shape[0] * len(test_generator)

print("Total Records (n): {}".format(df.count()))
print("Total Records after adjusting for 24 hours: {}".format(len(X)))
print("Number of samples in training set (.8 * n): trainX = {}".format(trainX.shape[0]))
print("Number of samples in testing set (.2 * n): testX = {}".format(testX.shape[0]))
print("Size of individual batches: {}".format(test_X.shape[1]))
print("Number of total samples in training feature set: {}".format(train_samples))
print("Number of samples in testing feature set: {}".format(test_samples))

# theres 2 features - day and count so set index to around 10 initially
units = 10
num_epoch = 1000
learning_rate = 0.00144

model = Sequential()
model.add(LSTM(units, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.1))
model.add(Dense(1))

adam = Adam(lr=learning_rate)
# Stop training when a monitored quantity has stopped improving.
# callback = [EarlyStopping(monitor="loss", min_delta = 0.00001, patience = 50, mode = 'auto', restore_best_weights=True), tensorboard]

# Using regression loss function 'Mean Standard Error' and validation metric 'Mean Absolute Error'
model.compile(loss='mse', optimizer=adam, metrics=['mae'])

# fit network
history = model.fit_generator(train_generator,
                              epochs=num_epoch,
                              validation_data=test_generator,
                              verbose=2,
                              shuffle=False,
                              initial_epoch=0)

# serialize model to JSON
model_json = model.to_json()
with open("best_fit_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("best_fit_model.h5")
print("Saved model to disk")

for key in history.history.keys():
    print(key)

# Calculate the train loss and train metric, in this case mean absolute error
train_loss = np.mean(history.history['loss'])
train_mae = np.mean(history.history['mae'])

val_loss = history.history['val_loss'][-1]
val_mae = history.history['val_mae'][-1]

print(val_loss)
print(val_mae)

title = 'Train Loss: {0:.3f} Test Loss: {1:.3f}\n  Train MAE: {2:.3f}, Val MAE: {3:.3f}'.format(train_loss, val_loss,
                                                                                                train_mae, val_mae)

# Plot loss function
fig = pyplot.figure()
pyplot.style.use('seaborn')

pyplot.plot(history.history['loss'], 'c-', label='train')
pyplot.plot(history.history['val_loss'], 'm:', label='test')
pyplot.text(epoch-2, 0.07, rmse , style='italic')
pyplot.title(title)
pyplot.legend()
pyplot.grid(True)
fig.set_size_inches(w=7, h=7)
pyplot.close()
pyplot.show()

# load json and create model
json_file = open('best_fit_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("best_fit_model.h5")
print("Loaded model from disk")

yhat_train_plot = loaded_model.predict(train_X)
yhat_test_plot = loaded_model.predict(test_X)


"""
New code
"""
# Generate new dates for the predictions
date_list = [base_timestamp + timedelta(minutes=x+3) for x in range(0, len(yhat_test_plot)*3, 3)]
# Generate weekdays from the above dates (and transform them using the label encoder used above)
weekday_list = encoder.transform(np.array([x.strftime("%A")[:3] for x in date_list]))
# Concatenate the predictions with the transformated weekday list
prediction_weekday = np.hstack((yhat_test_plot, weekday_list.reshape(-1, 1)))

# Using the scaler using to transform input data, inverse transform the above array to get
# predictions in the original scale
predictions = scaler.inverse_transform(prediction_weekday)
# Make a dataframe out of the resultant array and fix dates and weekdays
prediction_df = DataFrame(predictions, columns=["predictions", "day"])
prediction_df["timestamp"] = date_list
prediction_df["day"] = encoder.inverse_transform(weekday_list)

print("Predictions:")
print(prediction_df.head())


"""
New code ends
"""

print(train_X.shape)
print(yhat_train_plot.shape)

print(test_X.shape)
print(yhat_test_plot.shape)

fig = pyplot.figure()
pyplot.style.use('seaborn')
palette = pyplot.get_cmap('Set1')
# pyplot.plot(y[:, n_lead-1], marker='', color=palette(4), linewidth=1, alpha=0.9, label='actual')
pyplot.plot(y, marker='', color=palette(4), linewidth=1, alpha=0.9, label='actual')
pyplot.plot(yhat_train_plot, marker='', color=palette(2), linewidth=1, alpha=0.9, label='training predictions')
pyplot.plot(yhat_test_plot, marker='', color=palette(3), linewidth=1, alpha=0.9, label='testing predictions')

pyplot.title('Crowd Prediction', loc='center', fontsize=20, fontweight=5, color='orange')
pyplot.ylabel('Count')
pyplot.legend()
fig.set_size_inches(w=15, h=5)
pyplot.show()
