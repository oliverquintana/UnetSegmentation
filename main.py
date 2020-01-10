import matplotlib.pyplot as plt
from model import *
from get_data import *

# Load dataset
x_train, y_train, x_dev, y_dev = load_data()

# Data normalization
x_train = x_train.astype('float32') / 255.
x_dev = x_test.astype('float32') / 255.
y_train = y_train.astype('float32') / 255.
y_dev = y_test.astype('float32') / 255.

# Load Unet model
model = unet()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mse'])

# Show model summary
model.summary()

# Train model
model.fit(x_train, y_train,
          epochs = 10,
          batch_size = 32,
          shuffle = True,
          validation_data = (x_dev, y_dev))

decoded_imgs = model.predict(x_test)
model.save_weights('newWeights.h5')

# Display results
show_predictions()
