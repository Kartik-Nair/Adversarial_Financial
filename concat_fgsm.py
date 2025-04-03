import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tqdm import tqdm
from tensorflow.keras.losses import sparse_categorical_crossentropy
import matplotlib.pyplot as plt


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


def concat_fgsm_attack_seq(model, X, y, epsilon=0.1, num_steps=30, num_random_transactions=5, random_seed=42):
    """
    Concat FGSM [seq] - Sequential version with proper variable handling
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    batch_size, seq_length, feature_dim = X.shape
    feature_range = (-3.0, 3.0)  # For standardized data
    
    # Initialize with original + space for additions
    X_adv = np.concatenate([
        X,
        np.random.uniform(*feature_range, size=(batch_size, num_random_transactions, feature_dim))
    ], axis=1)
    
    # Convert to Variable
    X_adv_var = tf.Variable(X_adv, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    
    # Add and perturb one transaction at a time
    for k in range(num_random_transactions):
        pos = seq_length + k  # Position to perturb
        
        for _ in tqdm(range(num_steps)):
            with tf.GradientTape() as tape:
                predictions = model(X_adv_var)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y_tensor, predictions))
            
            gradients = tape.gradient(loss, X_adv_var)
            
            if gradients is None:
                raise ValueError("Gradients are None - check model architecture")
            
            # Create updated tensor
            perturbation = epsilon * tf.sign(gradients[:, pos, :])
            new_values = X_adv_var[:, pos, :] + perturbation
            new_values = tf.clip_by_value(new_values, *feature_range)
            
            # Update using variable assignment
            X_adv_var[:, pos, :].assign(new_values)
    
    return X_adv_var.numpy()

def concat_fgsm_attack_sim(model, X, y, epsilon=0.1, num_steps=30, num_random_transactions=5, random_seed=42):
    """
    Concat FGSM [sim] - Simultaneous version with proper variable handling
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    batch_size, seq_length, feature_dim = X.shape
    feature_range = (-3.0, 3.0)
    
    # Initialize with original + random additions
    X_adv = np.concatenate([
        X,
        np.random.uniform(*feature_range, size=(batch_size, num_random_transactions, feature_dim))
    ], axis=1)
    
    X_adv_var = tf.Variable(X_adv, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    
    # Perturb all added transactions simultaneously
    for _ in tqdm(range(num_steps)):
        with tf.GradientTape() as tape:
            predictions = model(X_adv_var)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_tensor, predictions))
        
        gradients = tape.gradient(loss, X_adv_var)
        
        if gradients is None:
            raise ValueError("Gradients are None - check model architecture")
        
        # Update all added transactions
        perturbation = epsilon * tf.sign(gradients[:, seq_length:, :])
        new_values = X_adv_var[:, seq_length:, :] + perturbation
        new_values = tf.clip_by_value(new_values, *feature_range)
        
        # Variable assignment
        X_adv_var[:, seq_length:, :].assign(new_values)
    
    return X_adv_var.numpy()

def visualize_perturbations(original, adv_seq, adv_sim):
    """
    Visualizes the differences between original and adversarial examples
    for both sequential and simultaneous approaches.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Get a sample sequence (first in the batch)
    sample_idx = 0
    original_sample = original[sample_idx]
    adv_seq_sample = adv_seq[sample_idx]
    adv_sim_sample = adv_sim[sample_idx]
    
    # Plot original sequence
    axes[0].plot(original_sample[:, 0], label='Amount')
    axes[0].plot(original_sample[:, 1], label='MCC')
    axes[0].set_title('Original Transaction Sequence')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot sequential approach
    axes[1].plot(adv_seq_sample[:, 0], label='Amount')
    axes[1].plot(adv_seq_sample[:, 1], label='MCC')
    axes[1].axvline(x=original.shape[1]-0.5, color='r', linestyle='--', label='Original End')
    axes[1].set_title('Concat FGSM (Sequential)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot simultaneous approach
    axes[2].plot(adv_sim_sample[:, 0], label='Amount')
    axes[2].plot(adv_sim_sample[:, 1], label='MCC')
    axes[2].axvline(x=original.shape[1]-0.5, color='r', linestyle='--', label='Original End')
    axes[2].set_title('Concat FGSM (Simultaneous)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('concat_fgsm_comparison.png')
    plt.show()

X_train_tensor, y_train_tensor, train_lengths = preprocess_rosbank_data(df_train)

# Split the data into training and test sets
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_train_tensor, y_train_tensor, test_size=0.35, random_state=42)

X_test_tensor = X_test_tensor[:50]
y_test_tensor = y_test_tensor[:50]
# Create and train the model
input_shape = X_train_tensor.shape[1:]  # Shape of the input data
model = models.load_model('GRU_model.h5')  # Replace with your saved model path

# Evaluate the model on original test data
original_loss, original_accuracy = model.evaluate(X_test_tensor, y_test_tensor)
print(f"Original Test Accuracy: {original_accuracy * 100:.2f}%")

# Parameters for the attacks
epsilon = 1
num_steps = 30
num_random_transactions = 5

# Perform Concat FGSM (Sequential) attack
print("Running Concat FGSM (Sequential) attack...")
X_adv_seq = concat_fgsm_attack_seq(
    model=model,
    X=X_test_tensor,
    y=y_test_tensor,
    epsilon=epsilon,
    num_steps=num_steps,
    num_random_transactions=num_random_transactions
)

# Perform Concat FGSM (Simultaneous) attack
print("Running Concat FGSM (Simultaneous) attack...")
X_adv_sim = concat_fgsm_attack_sim(
    model=model,
    X=X_test_tensor,
    y=y_test_tensor,
    epsilon=epsilon,
    num_steps=num_steps,
    num_random_transactions=num_random_transactions
)


# Evaluate the model on Concat FGSM adversarial examples
seq_loss, seq_accuracy = model.evaluate(X_adv_seq, y_test_tensor)
sim_loss, sim_accuracy = model.evaluate(X_adv_sim, y_test_tensor)

print(f"Concat FGSM [seq] Test Accuracy: {seq_accuracy * 100:.2f}%")
print(f"Concat FGSM [sim] Test Accuracy: {sim_accuracy * 100:.2f}%")

# Visualize the perturbations
visualize_perturbations(X_test_tensor, X_adv_seq, X_adv_sim)