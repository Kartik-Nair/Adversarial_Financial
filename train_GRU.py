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
from torch.utils.data import DataLoader, TensorDataset

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

    # Group transactions by user (assuming ⁠ cl_id⁠  as user ID) and treat each user's transactions as a sequence
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
train_data = train_dataset['train']
df_train = pd.DataFrame(train_data)

# Simple 65/35 split as requested
train_df, test_df = train_test_split(df_train, test_size=0.35, random_state=42)

# Preprocess each dataset
X_train_tensor, y_train_tensor = preprocess_rosbank_data(train_df)
X_test_tensor, y_test_tensor = preprocess_rosbank_data(test_df)

# Create DataLoaders for batched training
batch_size = 1024
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Bidirectional GRU Model as per the paper specification
class BidirectionalGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim=256, dropout_rate=0.1):
        super(BidirectionalGRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size, 
            hidden_dim, 
            batch_first=True,
            bidirectional=True,  # Make GRU bidirectional as per paper
            num_layers=1         # Paper specifies 1 layer
        )
        self.dropout = nn.Dropout(dropout_rate)  # Dropout of 0.1 as per paper
        self.fc = nn.Linear(hidden_dim * 2, 2)  # *2 because bidirectional concatenates forward and backward passes
        
    def forward(self, x):
        # GRU returns output and final hidden state
        # For bidirectional GRU, output is (batch, seq_len, hidden_dim*2)
        output, _ = self.gru(x)
        
        # Get the last time step output from both directions
        # Forward direction: output[:, -1, :hidden_dim]
        # Backward direction: output[:, 0, hidden_dim:]
        
        # Alternative approach: use the output from the last time step
        # which contains information from both directions
        last_output = output[:, -1, :]
        
        # Apply dropout as per paper
        last_output = self.dropout(last_output)
        
        # Pass through final fully connected layer
        logits = self.fc(last_output)
        
        return logits

# Initialize the Bidirectional GRU model
input_size = 2  # Amount and Transaction Category (2 features)
model = BidirectionalGRUClassifier(input_size, hidden_dim=256, dropout_rate=0.1)

# Initialize optimizer (Adam with step size 0.001 as per paper)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluation function to compute accuracy and loss
def evaluate_model(model, data_loader):
    """
    Evaluate the model on the given dataset.
    
    Args:
        model: The trained model.
        data_loader: DataLoader for the evaluation data.
    
    Returns:
        accuracy: Accuracy of the model on the dataset.
        loss: Loss of the model on the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for sequences, targets in data_loader:
            # Forward pass
            outputs = model(sequences)
            loss = loss_function(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)  # Multiply by batch size for weighted average
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct_predictions / total
    avg_loss = total_loss / total
    
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

# Training Loop with early stopping based on test loss
def train_model(model, train_loader, test_loader, optimizer, num_epochs=50):
    """
    Train the GRU model with batched processing and early stopping.
    
    Args:
        model: The target model.
        train_loader: DataLoader for the training data.
        test_loader: DataLoader for the test data.
        optimizer: The optimizer to use for training.
        num_epochs: Maximum number of epochs to train (50 as per paper).
    
    Returns:
        model: The trained model.
    """
    # Early stopping parameters
    best_test_loss = float('inf')
    patience = 3  # Stop if no improvement for 3 epochs as per paper
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        total_samples = 0
        
        # Use tqdm to add a progress bar to the training loop
        for sequences, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            # Zero gradients from previous step
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sequences)
            loss = loss_function(outputs, targets)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / total_samples
        
        # Evaluate on training set
        train_accuracy, train_detailed_loss = evaluate_model(model, train_loader)
        
        # Evaluate on test set
        test_accuracy, test_loss = evaluate_model(model, test_loader)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Check for early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save the best model
            save_model(model, f"best_GRU_epoch_{epoch+1}.pth")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_accuracy, test_loss = evaluate_model(model, test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")
    
    return model

# Train the model with batching and early stopping
print("Starting training...")
model = train_model(model, train_loader, test_loader, optimizer, num_epochs=50)