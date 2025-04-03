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
    Performs the Concat FGSM attack by adding random transactions to the end of the original sequence
    and then applying FGSM only on these added transactions.
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    batch_size, seq_length, feature_dim = X.shape
    
    # Create a copy of the original data
    X_extended = np.zeros((batch_size, seq_length + num_random_transactions, feature_dim))
    X_extended[:, :seq_length, :] = X  # Copy original sequences
    
    # Initialize random transactions at the end
    feature_mins = np.min(X, axis=(0, 1))
    feature_maxs = np.max(X, axis=(0, 1))
    
    for i in range(batch_size):
        for j in range(num_random_transactions):
            for k in range(feature_dim):
                X_extended[i, seq_length + j, k] = np.random.uniform(
                    feature_mins[k], feature_maxs[k]
                )
    
    # Convert to a trainable Variable
    X_extended_tensor = tf.Variable(X_extended, dtype=tf.float32, trainable=True)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    
    for step in tqdm(range(num_steps)):
        with tf.GradientTape() as tape:
            # Get predictions (using the variable's value)
            predictions = model(X_extended_tensor.value())
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_tensor, predictions)
        
        # Compute gradients w.r.t. the entire tensor
        gradients = tape.gradient(loss, X_extended_tensor)
        
        if gradients is None:
            raise ValueError("Gradients are None! Check model architecture and input types.")
        
        # Only perturb the added transactions
        added_gradients = gradients[:, seq_length:, :]
        perturbations = tf.sign(added_gradients)
        
        # Update only the added part
        updated_added = X_extended_tensor[:, seq_length:, :] + epsilon * perturbations
        updated_added = tf.clip_by_value(
            updated_added, 
            feature_mins, 
            feature_maxs
        )
        
        # Assign back using tf.Variable's assign method
        X_extended_tensor[:, seq_length:, :].assign(updated_added)
    
    return X_extended_tensor.numpy()



def concat_fgsm_attack_sim(model, X, y, epsilon=0.1, num_steps=30, num_random_transactions=5, random_seed=42):
    """
    Performs the Concat FGSM (simultaneous) attack by first adding random transactions
    and then applying FGSM to the entire sequence simultaneously.
    
    Args:
        model: The trained TensorFlow model.
        X: The original transaction sequences.
        y: The true labels for the sequences.
        epsilon: The magnitude of the perturbation.
        num_steps: Number of steps for perturbation.
        num_random_transactions: Number of random transactions to add to each sequence.
        random_seed: Random seed for reproducibility.
        
    Returns:
        X_adv: The adversarial examples with added and perturbed transactions.
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    batch_size, seq_length, feature_dim = X.shape
    
    # Create a copy of the original data with extended length
    X_extended = np.zeros((batch_size, seq_length + num_random_transactions, feature_dim))
    
    # Copy original sequences
    X_extended[:, :seq_length, :] = X
    
    # Initialize random transactions at the end
    feature_mins = np.min(X, axis=(0, 1))
    feature_maxs = np.max(X, axis=(0, 1))
    
    for i in range(batch_size):
        for j in range(num_random_transactions):
            for k in range(feature_dim):
                X_extended[i, seq_length + j, k] = np.random.uniform(
                    feature_mins[k], feature_maxs[k]
                )
    
    # Convert to tensor for gradient computation
    X_extended_tensor = tf.convert_to_tensor(X_extended, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    
    # Apply FGSM to the entire sequence (including both original and added transactions)
    for _ in tqdm(range(num_steps)):
        with tf.GradientTape() as tape:
            # Watch the entire sequence
            tape.watch(X_extended_tensor)
            
            # Get the model's prediction
            predictions = model(X_extended_tensor)
            loss = sparse_categorical_crossentropy(y_tensor, predictions)
        
        # Calculate gradients
        gradients = tape.gradient(loss, X_extended_tensor)
        
        # Get the sign of the gradients
        perturbations = tf.sign(gradients)
        
        # Update the entire sequence
        X_extended_tensor = X_extended_tensor + epsilon * perturbations
        
        # Clip if necessary
        X_extended_tensor = tf.clip_by_value(
            X_extended_tensor, 
            tf.convert_to_tensor(feature_mins, dtype=tf.float32),
            tf.convert_to_tensor(feature_maxs, dtype=tf.float32)
        )
    
    return X_extended_tensor.numpy()

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