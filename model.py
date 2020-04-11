from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np

OPTIMAL_HIDDEN_NEURONS_COUNT = 38
FILE_NAME = 'model.h5'


class Model:

    def __init__(self, hidden_neurons=OPTIMAL_HIDDEN_NEURONS_COUNT, load=False):
        """If load==True, load model from file, else create new model."""
        if load:
            self.model = load_model(FILE_NAME)
        else:
            self.model = Sequential()
            self.model.add(Dense(hidden_neurons, input_dim=3, activation='tanh'))
            self.model.add(Dense(2, activation='linear'))
            self.model.compile(optimizer='rmsprop', loss='mse')

    def train(self, epochs, batch_size, validation_split, verbose=2):
        """Train model and return list of loss errors."""
        inputs = 2 * np.random.rand(epochs, 3) - 1
        goals = np.zeros((epochs, 2))

        for i in range(epochs):
            goals[i, 0] = max(inputs[i, :]) * inputs[i, 1]
            goals[i, 1] = inputs[i, 0] ** 2 - inputs[i, 1] * inputs[i, 2]

        hist = self.model.fit(
            x=inputs,
            y=goals,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

        return hist.history['loss']

    def test(self, size):
        """Test model with random inputs."""
        inputs = 2 * np.random.rand(size, 3) - 1
        goals = np.zeros((size, 2))
        outputs = np.zeros((size, 2))

        for i in range(size):
            goals[i, 0] = max(inputs[i, :]) * inputs[i, 1]
            goals[i, 1] = inputs[i, 0] ** 2 - inputs[i, 1] * inputs[i, 2]
            outputs[i, :] = self.model.predict(np.array([[inputs[i, 0], inputs[i, 1], inputs[i, 2]]]))

        return inputs, outputs, goals

    def save(self):
        """Save model to file."""
        self.model.save(FILE_NAME)

    def summary(self):
        """Show summary of model."""
        self.model.summary()

    def predict(self, inputs):
        """Predict value from inputs."""
        return self.model.predict(inputs)
