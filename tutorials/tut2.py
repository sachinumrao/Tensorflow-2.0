import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Read data into the dataframe
x = pd.read_csv("~/Data/Kaggle/titanic/processed_train.csv")
x.head()

df = pd.read_csv("~/Data/Kaggle/titanic/train.csv")
y = df['Survived']
y.head()

subm_data = pd.read_csv("~/Data/Kaggle/titanic/processed_test.csv")
subm_data.head()

# Load dataset in tf.data
x_train, x_val, y_train, y_val = train_test_split(x.values, y.values,
                                                  test_size=0.20,
                                                  random_state=99,
                                                  shuffle=True)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Create linear mdoel
model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape=(22,), units=128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=20)

# Evaluate model
model.evaluate(test_dataset)

# Score model on submission data
subm_dataset = tf.data.Dataset.from_tensor_slices(
                (subm_data.values)).batch(subm_data.shape[0])

y_subm = model.predict(subm_dataset)
threshold = 0.4
y_subm = (y_subm > threshold).astype('int')
out_df = pd.read_csv("~/Data/Kaggle/titanic/subm.csv")

out_df['Survived'] = y_subm

out_df.to_csv("~/Data/Kaggle/titanic/subm_keras.csv", index=False)
