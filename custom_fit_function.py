import numpy as np
import tensorflow as tf
from tensorflow import keras
keras.backend.clear_session()
print(tf.__version__)


class CustomModel(keras.Model):

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute loss
            loss = self.compiled_loss(y, y_pred,
                                      regularization_losses=self.losses)
            # Measure gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradients(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update loss metric
            self.compiled_metrics.update(y, y_pred)

            return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred,
                           regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


# Create model
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Create dummy train data
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))

# Train the model
print("\nTraining the model...")
model.fit(x, y, epochs=5)

# Create dummy test data
x_test = np.random.random((100, 32))
y_test = np.random.random((100, 1))

# Evaluate the model
print("\nTesting the model...")
model.evaluate(x_test, y_test)

# Testing same implementation with data generators
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(32).repeat(10)

# Reinstantiate the model
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nFitting model with data generator using batch size...")
model.fit(dataset, epochs=10)
