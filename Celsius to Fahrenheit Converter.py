import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

temperature = float(input('Insert the temperature in Celsius you with to convert: '))

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

hidden1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hidden2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)

# Units: Number of neurons in the output layer
# Input_shape: Number of neurons in the input layer

model = tf.keras.Sequential([hidden1, hidden2, output])
# (There are several models, and allow us to work with layers.)

# We compile the model, preparing it for learning:
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),  
    loss = 'mean_squared_error'
)
# We define an optimizer (Algoritmo Adam) with a learning rate
# and set up the loss function (mean squared error)

print('Starting the learning process... ')
historial = model.fit(celsius,fahrenheit, epochs=1000, verbose=False)
# To train the model we use the function "fit", indicating the predictors, the target
# and the number of epochs. We also 'mute' the prints during training.
print('Model trained!')

# We can plot the error function:
plt.xlabel("# Epoch")
plt.ylabel("Loss magnitude")
plt.plot(historial.history["loss"])
plt.show()


print('\nThe internal variables of the model are: ')
print(hidden1.get_weights())
print(hidden2.get_weights())
print(output.get_weights())
print('Where the first values are the weights, and the second the biases.')


# We are now able to make predictions, e.g.:
result = model.predict([temperature])
print('\nThe prediction for the temperature is', round(np.float64(result)), 'Fahrenheit.')