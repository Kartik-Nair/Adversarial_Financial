import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

# Assuming CrossEntropyLoss as the loss function
loss_function = nn.CrossEntropyLoss()

def preprocess_rosbank_data(df):
    """
    Preprocess the Rosbank dataset: Normalize numerical features, one-hot encode categorical ones, and 
    group transactions into sequences (e.g., for each user or session).
    
    Args:
        df (pd.DataFrame): The raw transaction data.
        
    Returns:
        X_train_tensor (torch.Tensor): Processed features as PyTorch tensor.
        y_train_tensor (torch.Tensor): Target labels as PyTorch tensor.
    """
    # Handle missing data if any
    df = df.dropna(subset=['amount', 'MCC', 'target_flag'])  # Drop rows with missing values in important columns

    label_encoder = LabelEncoder()
    df['MCC'] = label_encoder.fit_transform(df['MCC']) 

    # Normalize continuous features like 'amount'
    scaler = StandardScaler()
    df['amount'] = scaler.fit_transform(df[['amount']]) 

    # Group transactions by user (assuming ⁠ cl_id⁠  as user ID) and treat each user’s transactions as a sequence
    grouped = df.groupby('cl_id')

    # Convert the features and target to lists of tensors (for sequence input to RNN)
    X_sequences = []
    y_sequences = []
    for _, group in grouped:
        # Convert the group (user's transactions) to a tensor of features (scaled amount, encoded MCC)
        X_sequences.append(torch.tensor(group[['amount', 'MCC']].values, dtype=torch.float32))  
        
        # We take the last target flag of the group for classification
        y_sequences.append(torch.tensor(group["target_flag"].values[-1], dtype=torch.long)) 
    
    # Pad sequences to ensure uniform length across the batch
    padded_X_sequences = pad_sequence(X_sequences, batch_first=True, padding_value=0)  # Padding with zeros
    padded_y_sequences = torch.tensor(y_sequences, dtype=torch.long)

    return padded_X_sequences, padded_y_sequences


# Example: Load the dataset from Hugging Face
train_dataset = load_dataset("dllllb/rosbank-churn", "train")
print(train_dataset)
train_data = train_dataset['train']
df_train = pd.DataFrame(train_data)
train_df, test_df = train_test_split(df_train, test_size=0.35, random_state=42)

# Preprocess the dataset to create sequential data
X_all_tensor, y_all_tensor = preprocess_rosbank_data(df_train)

# Now, split the preprocessed data into 65% training data and 35% test data (with labels)
# Set a random seed for reproducibility
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(
    X_all_tensor, y_all_tensor, test_size=0.35, random_state=42
)

# Define GRU Model (as per the paper)
class GRUClassifier(nn.Module):
    def __init__(self, input_size, embed_dim=128, hidden_dim=256):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification (0 or 1)
        
    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])  # Use the last hidden state for classification
        return out

# Initialize the GRU model (input_size is 2, since we have two features: amount, MCC)
input_size = 2  # Amount and Transaction Category (2 features)
model = GRUClassifier(input_size)

# Initialize optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluation function to compute test accuracy and loss
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The trained model.
        X_test: Features of the test set (sequences).
        y_test: Target labels for the test set.
    
    Returns:
        accuracy: Accuracy of the model on the test set.
        loss: Loss of the model on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct_predictions = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for i in range(len(X_test)):
            sequence = X_test[i]
            target_label = y_test[i].unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            output = model(sequence.unsqueeze(0))  # Add batch dimension for a single sample
            loss = loss_function(output, target_label)
            
            test_loss += loss.item()  # Accumulate loss
            
            # Get predictions
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target_label).sum().item()
            total += target_label.size(0)
    
    accuracy = correct_predictions / total  # Accuracy on the test set
    avg_loss = test_loss / len(X_test)  # Average loss on the test set
    
    return accuracy, avg_loss

# Function to save the entire model
def save_model(model, file_name="model.pth"):
    """
    Save the entire model (architecture + parameters).
    
    Args:
        model: The trained model.
        file_name: Name of the model file to save.
    """
    torch.save(model, file_name)
    print(f"Model saved as {file_name}")

# Training Loop (with Test Accuracy and Loss after each epoch, and model checkpointing)
def train_model(model, X_train, y_train, X_test, y_test, optimizer, num_epochs=10):
    """
    Train the GRU model and save the entire model after each epoch.
    
    Args:
        model: The target model.
        X_train: Features of the training set (sequences).
        y_train: Target labels for the training set.
        X_test: Features of the test set (sequences).
        y_test: Target labels for the test set.
        optimizer: The optimizer to use for training.
        num_epochs: Number of epochs to train.
    
    Returns:
        model: The trained model.
    """
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Use tqdm to add a progress bar to the training loop
        for i in tqdm(range(len(X_train)), desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            sequence = X_train[i]
            target_label = y_train[i].unsqueeze(0)  # Add batch dimension

            # Zero gradients from previous step
            optimizer.zero_grad()

            # Forward pass
            output = model(sequence.unsqueeze(0))  # Add batch dimension for a single sample
            loss = loss_function(output, target_label)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item()
        
        # Print training loss after each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss/len(X_train)}")
        
        train_accuracy, train_loss = evaluate_model(model, X_train, y_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy*100:.2f}%, Train Loss: {train_loss:.4f}")
        # Evaluate on the test set after each epoch
        test_accuracy, test_loss = evaluate_model(model, X_test, y_test)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy*100:.2f}%, Test Loss: {test_loss:.4f}")
        
        # Save the entire model after each epoch
        save_model(model, f"model_epoch_{epoch+1}.pth")  # Save the model with the epoch number in the file name
    
    return model


# Train the model on original data first
model = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, optimizer, num_epochs=10)
