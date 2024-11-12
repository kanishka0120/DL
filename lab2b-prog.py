import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt

# 1. Create synthetic data with separate training and validation sets
def create_data():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    return X_train, y_train, X_val, y_val

# 2. Define a deep neural network with optional dropout and batch normalization
def create_model(input_shape=(10,), dropout_rate=0.2):
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=input_shape),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(20, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(1)
    ])
    return model

# 3. Train the model with validation, learning rate scheduler, and early stopping
def train_model_with_history(model, optimizer, X_train, y_train, X_val, y_val, 
                             batch_size, epochs, optimizer_name):
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=1, 
        callbacks=[early_stopping, lr_scheduler]
    )
    print(f"{optimizer_name} - Final training loss: {history.history['loss'][-1]:.4f}")
    return history

# 4. Compare performance of SGD and Adam optimizers
# Load data
X_train, y_train, X_val, y_val = create_data()

# Create models for SGD and Adam
model_sgd = create_model()
model_adam = create_model()

# Optimizers
optimizer_sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9)
optimizer_adam = optimizers.Adam(learning_rate=0.001)

# Set training parameters
epochs = 50
batch_size = 32

# Train models and capture history with early stopping and learning rate scheduling
print("\nTraining with SGD optimizer")
history_sgd = train_model_with_history(model_sgd, optimizer_sgd, X_train, y_train, X_val, y_val, batch_size, epochs, 'SGD')

print("\nTraining with Adam optimizer")
history_adam = train_model_with_history(model_adam, optimizer_adam, X_train, y_train, X_val, y_val, batch_size, epochs, 'Adam')

# 5. Plot the loss curves for comparison
plt.plot(history_sgd.history['loss'], label='SGD - Training Loss', color='blue')
plt.plot(history_sgd.history['val_loss'], label='SGD - Validation Loss', color='lightblue')
plt.plot(history_adam.history['loss'], label='Adam - Training Loss', color='orange')
plt.plot(history_adam.history['val_loss'], label='Adam - Validation Loss', color='peachpuff')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("SGD vs Adam Optimizer: Loss Comparison")
plt.legend()
plt.grid(True)
plt.show()