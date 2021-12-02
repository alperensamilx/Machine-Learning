import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataFrame = pd.read_excel('merc.xlsx')
# print(dataFrame.head())
# print(dataFrame.describe())

# print(dataFrame.isnull().sum())

newDf = dataFrame.sort_values("price", ascending=False).iloc[131:]

dataFrame = newDf

dataFrame = dataFrame[dataFrame.year != 1970]

# print(dataFrame.head())

dataFrame = dataFrame.drop("transmission", axis=1)

y = dataFrame["price"].values
x = dataFrame.drop("price", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=300)

loss_data = pd.DataFrame(model.history.history)

# loss_data.plot()
# plt.show()

predictSeries = model.predict(x_test)
print(mean_absolute_error(y_test, predictSeries))

newCarSeries = dataFrame.drop("price", axis=1).iloc[2]
newCarSeries = scaler.transform(newCarSeries.values.reshape(-1, 5))
print(dataFrame.iloc[2])
print(model.predict(newCarSeries))