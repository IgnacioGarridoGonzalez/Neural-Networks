import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel(r'C:\Users\Ignacio\Documents\PYTHON DOCUMENTS\concrete_data.xlsx')
print(df)

predictors = df.iloc[:,0:8]
print('\nPredictors: \n', predictors)

target = df.iloc[:,8]
print('\nTarget: \n', target)

n_cols = predictors.shape[1]
print(type(n_cols))

hidden1 = tf.keras.layers.Dense(units=5, activation='relu', input_shape=(n_cols,))
hidden2 = tf.keras.layers.Dense(units=5, activation='relu')
output = tf.keras.layers.Dense(units=1)

model = tf.keras.models.Sequential([hidden1, hidden2, output])
model.compile(
    optimizer = tf.keras.optimizers.Adam(),  
    loss = 'mean_squared_error'
)

print('Starting the learning process... ')
historial = model.fit(predictors, target, epochs=10000, verbose=True)
print('Model trained!')

plt.xlabel("# Epoch")
plt.ylabel("Loss magnitude")
plt.plot(historial.history["loss"])
plt.show()

sample = [600,0,0,162,2.5,1040,676,28]
sample_predict = np.array(sample)

# We are now able to make predictions, e.g.:
result = model.predict([sample])
print('The prediction for', sample,  'is', result, 'MPa.')

# For sample = [600,0,0,162,2.5,1040,676,28]
# epochs: 10000, loss = 19.3090, result: 98.500626 MPa

# For sample = [540,0,0,162,2.5,1047,676,28]
# epochs: 10000, loss = 31.4714, result: 70.93735 MPa
# epochs: 10000, loss = 4.1113e-04, result: 71.55524 MPa