# Import necessary libraries
# Dataset https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import os
#GPU OPTIMIZATION : Enable Mixed Precision for Faster Computation
mixed_precision.set_global_policy('mixed_float16') #gpu optimization 

# GPU OPTIMIZATION : Configure GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)
# Define model save path
MODEL_PATH = 'autoencoder_model.keras'
# Load and Preprocess the Dataset
data = pd.read_csv('creditcard_2023.csv')

# Check for NaN values and handle them
if data.isnull().values.any():
    print("Warning: NaN values found in the dataset. Handling missing values...")
    data = data.dropna()  # Alternatively, use imputation: data.fillna(data.mean())

# Separate normal and fraudulent transactions
normal = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Drop the class label for training
X_normal = normal.drop('Class', axis=1)
X_fraud = fraud.drop('Class', axis=1)

# Standardize the features
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_fraud_scaled = scaler.transform(X_fraud)

# Verify no NaN values remain after scaling
assert not np.any(np.isnan(X_normal_scaled)), "NaN values found in X_normal_scaled"
assert not np.any(np.isnan(X_fraud_scaled)), "NaN values found in X_fraud_scaled"

# Function to build the autoencoder model
def build_autoencoder(input_dim, encoding_dim=16):
    input_layer = layers.Input(shape=(input_dim,))
    # Encoder
    encoder = layers.Dense(64, activation='relu')(input_layer)
    encoder = layers.Dense(32, activation='relu')(encoder)
    encoder = layers.Dense(encoding_dim, activation='relu')(encoder) #bottleneck
    # Decoder
    decoder = layers.Dense(32, activation='relu')(encoder)
    decoder = layers.Dense(64, activation='relu')(decoder)
    decoder = layers.Dense(input_dim, activation='linear', dtype='float32')(decoder)
    # Autoencoder model
    autoencoder = models.Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Check if model exists, otherwise train and save it
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    autoencoder = tf.keras.models.load_model(MODEL_PATH)
else:
    print("No saved model found. Training new model...")
    # Convert to tf.data.Dataset
    full_dataset = tf.data.Dataset.from_tensor_slices(X_normal_scaled)
    num_samples = len(X_normal_scaled)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    # Shuffle and split the dataset
    full_dataset = full_dataset.shuffle(buffer_size=1024)
    train_dataset = full_dataset.take(train_size).batch(512).prefetch(tf.data.AUTOTUNE)
    val_dataset = full_dataset.skip(train_size).batch(512).prefetch(tf.data.AUTOTUNE)

    # Map the datasets to yield (x, x) for autoencoder training
    train_dataset = train_dataset.map(lambda x: (x, x))
    val_dataset = val_dataset.map(lambda x: (x, x))

    # Build and train the autoencoder
    input_dim = X_normal_scaled.shape[1]
    autoencoder = build_autoencoder(input_dim)
    history = autoencoder.fit(train_dataset,
                              epochs=100,
                              validation_data=val_dataset,
                              verbose=1)
    # Save the model
    autoencoder.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
# Step 7: Evaluate the Model
# Reconstruct normal and fraudulent transactions
normal_reconstructed = autoencoder.predict(X_normal_scaled, batch_size=512)
fraud_reconstructed = autoencoder.predict(X_fraud_scaled, batch_size=512)

# Calculate reconstruction error (MSE)
mse_normal = np.mean(np.square(X_normal_scaled - normal_reconstructed), axis=1)
mse_fraud = np.mean(np.square(X_fraud_scaled - fraud_reconstructed), axis=1)

# Set threshold as mean + 2*std of normal errors
threshold = np.mean(mse_normal) + 2 * np.std(mse_normal)

# Classify transactions: 1 if error > threshold (fraud), else 0 (normal)
y_pred_normal = (mse_normal > threshold).astype(int)
y_pred_fraud = (mse_fraud > threshold).astype(int)

# Combine predictions and true labels
y_true = np.concatenate([np.zeros(len(normal)), np.ones(len(fraud))])
y_pred = np.concatenate([y_pred_normal, y_pred_fraud])
y_scores = np.concatenate([mse_normal, mse_fraud])  # For AUC calculation

# Calculate evaluation metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_scores)

# Print results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC: {auc:.4f}')

#----------------------------TEST NEW TRANSACTION ----------------------------
# Test a Transaction
new_transaction = { # values from first line of dataset 
    'id': 123456,
    'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
    'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
    'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
    'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
    'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
    'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
    'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
    'Amount': 149.62
}

# Convert to DataFrame
new_transaction_df = pd.DataFrame([new_transaction])

# Scale the transaction using the same scaler
new_transaction_scaled = scaler.transform(new_transaction_df)

# Pass through the autoencoder
reconstructed = autoencoder.predict(new_transaction_scaled, verbose=1)

# Calculate reconstruction error (MSE)
mse = np.mean(np.square(new_transaction_scaled - reconstructed))

# Classify based on the threshold
is_fraud = mse > threshold
print(f"\nTesting New Transaction:")
print(f"Reconstruction Error (MSE): {mse:.6f}")
print(f"Threshold: {threshold:.6f}")
print(f"Classified as: {'Fraud' if is_fraud else 'Normal'}")
