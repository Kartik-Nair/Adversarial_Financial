import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# Load the dataset from HuggingFace
train_dataset = load_dataset("dllllb/rosbank-churn", "train")
train_data = train_dataset['train']
df_train = pd.DataFrame(train_data)

def preprocess_rosbank_data(df):
    """
    Preprocess the Rosbank dataset for TensorFlow: Normalize numerical features, one-hot encode categorical ones, 
    and group transactions into sequences (e.g., for each user or session).
    
    Args:
        df (pd.DataFrame): The raw transaction data.
        
    Returns:
        X_sequences (list of np.array): List of padded sequences of features.
        y_sequences (np.array): Corresponding labels.
        sequence_lengths (list): Sequence lengths for each user's transactions.
    """
    # Handle missing data if any
    df = df.dropna(subset=['amount', 'MCC', 'target_flag', 'TRDATETIME'])

    # Sort by user ID (cl_id) and timestamp (TRDATETIME) to ensure the transaction order for each user
    df['TRDATETIME'] = pd.to_datetime(df['TRDATETIME'], format='%d%b%y:%H:%M:%S')
    df = df.sort_values(by=['TRDATETIME'])  

    # Normalize the numerical columns ('amount' and 'MCC')
    scaler = StandardScaler()
    df[['amount', 'MCC']] = scaler.fit_transform(df[['amount', 'MCC']])

    # Group transactions by user (assuming 'cl_id' as user ID) and treat each user's transactions as a sequence
    grouped = df.groupby('cl_id')

    # Convert the features and target to lists of arrays (for sequence input to RNN)
    X_sequences = []
    y_sequences = []
    sequence_lengths = []

    for _, group in grouped:
        X_sequences.append(group[['amount', 'MCC']].values)  # Features
        y_sequences.append(group["target_flag"].values[-1])  # Target (last transaction)
        sequence_lengths.append(len(group))  # Sequence length (number of transactions)

    # Pad sequences to ensure uniform length across the batch
    padded_X_sequences = pad_sequences(X_sequences, padding='post', dtype='float32', value=0)

    y_sequences = np.array(y_sequences)

    return padded_X_sequences, y_sequences, sequence_lengths

# Define a Bidirectional LSTM model in TensorFlow
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=False), input_shape=input_shape))  # Bidirectional LSTM
    model.add(layers.Dropout(0.1))  # Dropout layer
    model.add(layers.Dense(2, activation='softmax'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Preprocess the data
X_train_tensor, y_train_tensor, train_lengths = preprocess_rosbank_data(df_train)

# Split the data into training and test sets
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_train_tensor, y_train_tensor, test_size=0.35, random_state=42)

# Create and train the model
input_shape = X_train_tensor.shape[1:]  # Shape of the input data

model = create_model(input_shape)

# Set up callbacks for EarlyStopping and ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Fit the model with callbacks
history = model.fit(
    X_train_tensor, y_train_tensor, 
    validation_data=(X_test_tensor, y_test_tensor),
    epochs=50, 
    batch_size=1024, 
    callbacks=[early_stopping, checkpoint]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_tensor, y_test_tensor)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
