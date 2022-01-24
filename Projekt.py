import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

df = pd.read_csv('data.csv', names=column_names)
df.head()
df.isna().sum()
df = df.iloc[1:, 1:]
df_norm = (df - df.mean()) / df.std()
df_norm.head()
y_mean = df['price'].mean()
y_std = df['price'].std()


def convert_label_value(pred):
    return int(pred * y_std + y_mean)  # trzeba cofnac normalizacje ceny


x = df_norm.iloc[:, :6]
x.head()
y = df_norm.iloc[:, -1]
y.head()
x_arr = x.values
y_arr = y.values
x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.05, random_state=0)


def get_model():
    model = Sequential([
        Dense(10, input_shape=(6,), activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1)
    ])
    model.compile(
        loss='mse',  # mean square error
        optimizer='adam'
    )
    return model


get_model().summary()
es_cb = EarlyStopping(monitor='val_loss', patience=5)

model=get_model()
preds_on_untrained = model.predict(x_test)

history = model.fit(
x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 100,
    callbacks = [es_cb]
)

plot_loss(history)
