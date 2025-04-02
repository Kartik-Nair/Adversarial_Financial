import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Tuple, Optional
import random
import torch.nn.functional as F

# Assuming you have a loss function defined (e.g., CrossEntropyLoss)
loss_function = nn.CrossEntropyLoss()


def get_model_and_target(dataset_name: str, model_type: str = "GRU") -> Tuple[nn.Module, Optional[nn.Module], int]:
    """
    Returns the appropriate model, substitute model (if applicable), 
    and the number of classes for a given dataset and model type.

    Args:
        dataset_name (str): The name of the dataset 
                            ("Age 1", "Age 2", "Leaving", or "Scoring").
        model_type (str, optional): The type of model to use 
                                   ("GRU", "LSTM", or "CNN"). 
                                   Defaults to "GRU".

    Returns:
        Tuple[nn.Module, nn.Module, int]: 
            -   The target model (nn.Module).
            -   The substitute model (nn.Module) or None if not applicable.
            -   The number of classes for the dataset (int).
    """
    if model_type not in ["GRU", "LSTM", "CNN"]:
        raise ValueError("Invalid model type. Choose from GRU, LSTM, or CNN.")

    # Define model architectures
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.1):
            super(GRUModel, self).__init__()
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.gru(x)
            x = x[:, -1, :]  # Take the last time step's output
            x = self.fc(x)
            return x

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.1):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Take the last time step's output
            x = self.fc(x)
            return x
    
    class CNNModel(nn.Module):
        def __init__(self, input_size, embedding_dim, num_classes, num_filters=64, kernel_size=3):
            super(CNNModel, self).__init__()
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.conv1d = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=1)  # Padding to keep sequence length
            self.relu = nn.ReLU()
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # Global max pooling
            self.fc = nn.Linear(num_filters, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = x.transpose(1, 2)  # Need (batch_size, embedding_dim, seq_len) for Conv1d
            x = self.conv1d(x)
            x = self.relu(x)
            x = self.global_max_pool(x).squeeze(-1) # Get the maximum value for each filter
            x = self.fc(x)
            return x

    # Model parameters (these should be tuned based on your data)
    input_size = 409  # Example, adjust based on your vocabulary size
    hidden_size = 256
    embedding_dim = 128 # For CNN
    num_classes = 2 # Default
    
    # Define models and number of classes based on the dataset
    if dataset_name == "Age 1":
        num_classes = 4
    elif dataset_name == "Age 2":
        num_classes = 4
    elif dataset_name == "Leaving":
        num_classes = 2
    elif dataset_name == "Scoring":
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize target model
    if model_type == "GRU":
        target_model = GRUModel(input_size, hidden_size, num_classes)
    elif model_type == "LSTM":
        target_model = LSTMModel(input_size, hidden_size, num_classes)
    elif model_type == "CNN":
        target_model = CNNModel(input_size, embedding_dim, num_classes)

    # Initialize substitute model (can be the same or different)
    # For simplicity, let's keep it the same for now, but you can modify it.
    substitute_model = None
    if model_type == "GRU":
        substitute_model = GRUModel(input_size, hidden_size, num_classes)
    elif model_type == "LSTM":
        substitute_model = LSTMModel(input_size, hidden_size, num_classes)
    elif model_type == "CNN":
        substitute_model = CNNModel(input_size, embedding_dim, num_classes)

    return target_model, substitute_model, num_classes

# Attack implementations (placeholders - complete these)

def mask_random_tokens(sequence, mask_prob=0.15, mask_token=103):
    """
    Randomly masks tokens in a sequence.

    Args:
        sequence (torch.Tensor): The input sequence of transaction codes.
        mask_prob (float): Probability of masking each token.
        mask_token (int): The token ID to use for masking.

    Returns:
        torch.Tensor: The sequence with some tokens masked.
    """
    masked_sequence = sequence.clone()
    for i in range(len(sequence)):
        if random.random() < mask_prob:
            masked_sequence[i] = mask_token
    return masked_sequence

def mlm_predict(mlm, tokenizer, masked_sequence):
    """
    Uses a Masked Language Model (MLM) to predict masked tokens.

    Args:
        mlm: The Masked Language Model.
        tokenizer: The tokenizer associated with the MLM.
        masked_sequence (torch.Tensor): The sequence with masked tokens.

    Returns:
        torch.Tensor: Predicted tokens for the masked positions.
    """
    # 1. Tokenize the masked sequence (if necessary for your MLM)
    #inputs = tokenizer(masked_sequence, return_tensors="pt") # Adjust based on your tokenizer

    # 2. Get MLM predictions
    with torch.no_grad():
        #outputs = mlm(**inputs) # Adjust based on your MLM's input requirements
        predictions = masked_sequence # Placeholder

    # 3. Extract predicted tokens for masked positions
    #masked_indices = (masked_sequence == tokenizer.mask_token_id).nonzero(as_tuple=True)[0] # Find indices of masked tokens
    #predicted_tokens = torch.argmax(predictions[0][masked_indices], dim=-1)
    predicted_tokens = masked_sequence
    return predicted_tokens

def replace_masked_tokens(masked_sequence, predicted_tokens, mask_token=103):
    """
    Replaces masked tokens in a sequence with predicted tokens.

    Args:
        masked_sequence (torch.Tensor): The sequence with masked tokens.
        predicted_tokens (torch.Tensor): The predicted tokens for the masked positions.
        mask_token (int): The token ID used for masking.

    Returns:
        torch.Tensor: The new sequence with masked tokens replaced.
    """
    new_sequence = masked_sequence.clone()
    masked_indices = (masked_sequence == mask_token).nonzero(as_tuple=True)[0]

    for i, index in enumerate(masked_indices):
        new_sequence[index] = predicted_tokens[i]
    return new_sequence

def select_best_adversarial(model, adversarial_outputs, adversarial_sequences, target_label):
    """
    Selects the best adversarial sequence based on model output.

    Args:
        model: The target model.
        adversarial_outputs (list): List of model outputs for each adversarial sequence.
        adversarial_sequences (list): List of adversarial sequences.
        target_label: The target class label.

    Returns:
        torch.Tensor: The best adversarial sequence.
    """
    best_sequence = None
    best_probability = float('inf')  # Initialize with a large value

    for i, outputs in enumerate(adversarial_outputs):
        # Assuming outputs are logits, you might need to apply softmax
        probabilities = torch.softmax(outputs, dim=-1)
        # target_class_probability = probabilities[0, target_label].item()  # Probability of the target class
        target_class_probability = torch.max(probabilities, dim=1)[0].item()

        if target_class_probability < best_probability:
            best_probability = target_class_probability
            best_sequence = adversarial_sequences[i]

    return best_sequence

def calculate_perplexity(language_model, tokenizer, sequence):
    """
    Calculates the perplexity of a sequence using a language model.

    Args:
        language_model: The language model.
        tokenizer: The tokenizer for the language model.
        sequence (torch.Tensor): The input sequence.

    Returns:
        float: The perplexity of the sequence.
    """
    # 1. Tokenize the sequence
    #inputs = tokenizer(sequence, return_tensors="pt") # Adjust based on your tokenizer

    # 2. Get language model output
    with torch.no_grad():
        #outputs = language_model(**inputs, labels=inputs["input_ids"]) # Adjust based on your LM
        outputs = sequence

    # 3. Calculate perplexity
    #loss = outputs.loss
    #perplexity = torch.exp(loss)
    perplexity = 1.0
    return perplexity #perplexity.item()

def sample_token_minimize_target_prob(model, sequence, probabilities, target_label, temperature=1.0):
    """
    Samples a token to minimize the target class probability.

    Args:
        model: The target model.
        sequence (torch.Tensor): The sequence to modify.
        probabilities (torch.Tensor): Probabilities for each token in the vocabulary.
        target_label: The target class label.
        temperature: Temperature for sampling.

    Returns:
        int: The sampled token index.
    """
    sampled_token = None
    min_prob = float('inf')

    # Sample multiple tokens and evaluate which one minimizes the target probability
    num_samples = 10  # You can adjust the number of samples

    for _ in range(num_samples):
        # 1. Sample a token
        sampled_token_candidate = torch.multinomial(F.softmax(probabilities / temperature, dim=-1), 1).item()

        # 2. Create a modified sequence
        modified_sequence = sequence.clone()
        modified_sequence[-1] = sampled_token_candidate

        # 3. Get the model's output
        with torch.no_grad():
            outputs = model(modified_sequence.unsqueeze(0))  # Add batch dimension

        # Assuming outputs are logits, you might need to apply softmax
        probabilities_modified = torch.softmax(outputs, dim=-1)
        target_prob = probabilities_modified[0, target_label].item()

        # 4. Check if this sampled token minimizes the probability
        if target_prob < min_prob:
            min_prob = target_prob
            sampled_token = sampled_token_candidate

    return sampled_token

def sampling_fool_attack(model, tokenizer, mlm, sequence, num_generated_sequences=100):
    """
    Generates adversarial examples by replacing tokens using a Masked Language Model.

    Args:
        model: The target model to attack.
        tokenizer: Tokenizer for the model.
        mlm: Masked Language Model.
        sequence: The input sequence of transaction codes.
        num_generated_sequences: Number of adversarial sequences to generate.

    Returns:
        best_adversarial_sequence: The adversarial sequence that maximizes attack success.
    """
    adversarial_sequences = []

    for _ in range(num_generated_sequences):
        # 1. Randomly mask some tokens in the input sequence
        masked_sequence = mask_random_tokens(sequence)  # Implement this

        # 2. Use MLM to predict replacement tokens
        predicted_tokens = mlm_predict(mlm, tokenizer, masked_sequence)  # Implement this

        # 3. Create a new adversarial sequence with replacements
        new_sequence = replace_masked_tokens(masked_sequence, predicted_tokens)  # Implement this
        adversarial_sequences.append(new_sequence)

    # 4. Evaluate each adversarial sequence on the model
    adversarial_outputs = [model(seq.unsqueeze(0)) for seq in adversarial_sequences] # Add batch dimension to seq

    # 5. Select the "most adversarial" sequence (e.g., lowest target class probability)
    best_adversarial_sequence = select_best_adversarial(model, adversarial_outputs, adversarial_sequences, target_label)  # Implement this

    return best_adversarial_sequence

def fgsm_attack(model, sequence, target_label, epsilon=0.01):
    """
    Generates an adversarial example by perturbing token embeddings using FGSM.

    Args:
        model: The target model.
        sequence: The input sequence of transaction codes.
        epsilon: Perturbation strength.

    Returns:
        adversarial_sequence: The generated adversarial sequence.
    """

    # Assuming 'sequence' is a tensor of token indices
    # Get the embedding for the sequence
    sequence_embedding = model.embedding(sequence)    
    # Ensure the embedding is a leaf variable and we track gradients
    sequence_embedding.requires_grad = True

    # 1. Select a random token index
    token_index_to_perturb = random.randint(0, len(sequence) - 1)

    # 2. Get the embedding of the token
    sequence_embedding = model.embedding(sequence[token_index_to_perturb])
    sequence_embedding.requires_grad = True

    # 3. Forward pass
    outputs = model(sequence.unsqueeze(0)) # Ensure your model outputs logits # Add batch dimension
    loss = loss_function(outputs, target_label) # You'll need a target label

    # 4. Calculate gradients
    model.zero_grad()
    loss.backward()
    grad = token_embedding.grad

    # 5. Perturb the embedding
    perturbed_embedding = token_embedding + epsilon * grad.sign()

    # 6. Find the closest token in the embedding matrix
    embedding_matrix = model.embedding.weight.data # Get the embedding matrix
    distances = torch.cdist(perturbed_embedding.unsqueeze(0), embedding_matrix)
    closest_token_index = torch.argmin(distances)

    # 7. Create the adversarial sequence
    adversarial_sequence = sequence.clone()
    adversarial_sequence[token_index_to_perturb] = closest_token_index

    return adversarial_sequence

def concat_fgsm_attack(model, sequence, target_label, k=2, sequential=True, epsilon=0.01):
    """
    Generates an adversarial example by concatenating k transactions and applying FGSM.

    Args:
        model: The target model.
        sequence: The original sequence of transaction codes.
        k: The number of transactions to add.
        sequential: Whether to add transactions sequentially or simultaneously.
        epsilon: Perturbation strength.

    Returns:
        adversarial_sequence: The generated adversarial sequence.
    """
    if sequential:
        adversarial_sequence = sequence.clone()
        for _ in range(k):
            # 1. Add a random transaction
            random_transaction = torch.randint(0, 409, (1,))  # Assuming vocab_size is your vocabulary size
            adversarial_sequence = torch.cat((adversarial_sequence, random_transaction), dim=0)

            # 2. Apply FGSM to the last added transaction
            adversarial_sequence = fgsm_attack(model, adversarial_sequence, target_label, epsilon) # Reuse your FGSM function
    else:
        # 1. Add k random transactions
        random_transactions = torch.randint(0, 409, (k,))
        adversarial_sequence = torch.cat((sequence, random_transactions), dim=0)

        # 2. Apply FGSM to the added transactions
        adversarial_sequence = fgsm_attack(model, adversarial_sequence, target_label, epsilon) # Reuse your FGSM function

    return adversarial_sequence

def lm_fgsm_attack(model, language_model, tokenizer, sequence, target_label, perplexity_threshold, epsilon=0.01):
    """
    Generates adversarial examples using a language model and FGSM.

    Args:
        model: The target model.
        language_model: A pre-trained language model.
        tokenizer: Tokenizer for language model.
        sequence: The input sequence of transaction codes.
        target_label: The target label.
        perplexity_threshold: Maximum allowed perplexity for adversarial examples.
        epsilon: Perturbation strength for FGSM.

    Returns:
        adversarial_sequence: The generated adversarial sequence.
    """
    # 1. Select a token to perturb (e.g., randomly)
    token_index_to_perturb = random.randint(0, len(sequence) - 1)

    # 2. Get the embedding of the token
    token_embedding = model.embedding(sequence[token_index_to_perturb])
    token_embedding.requires_grad = True

    # 3. Forward pass
    outputs = model(sequence.unsqueeze(0))
    loss = loss_function(outputs, target_label)

    # 4. Calculate gradients
    model.zero_grad()
    loss.backward()
    grad = token_embedding.grad

    # 5. Perturb the embedding
    perturbed_embedding = token_embedding + epsilon * grad.sign()

    # 6. Find the closest token in the embedding matrix
    embedding_matrix = model.embedding.weight.data
    distances = torch.cdist(perturbed_embedding.unsqueeze(0), embedding_matrix)
    closest_token_index = torch.argmin(distances)

    # 7. Create a candidate adversarial sequence
    candidate_adversarial_sequence = sequence.clone()
    candidate_adversarial_sequence[token_index_to_perturb] = closest_token_index

    # 8. Calculate perplexity using the language model
    perplexity = calculate_perplexity(language_model, tokenizer, candidate_adversarial_sequence) # Implement this

    # 9. If perplexity is below the threshold, accept the sequence, otherwise, try another perturbation
    if perplexity < perplexity_threshold:
        adversarial_sequence = candidate_adversarial_sequence
    else:
        adversarial_sequence = sequence # Or you could recursively call the function or try a different token

    return adversarial_sequence

def concat_sampling_fool_attack(model, bert_model, tokenizer, sequence, target_label, k=2, temperature=1.0):
    """
    Generates adversarial examples by concatenating tokens and sampling with BERT.

    Args:
        model: The target model.
        bert_model: A pre-trained BERT model.
        tokenizer: Tokenizer for BERT.
        sequence: The original sequence of transaction codes.
        target_label: The target label.
        k: The number of transactions to add.
        temperature: Temperature for sampling.

    Returns:
        best_adversarial_sequence: The generated adversarial sequence.
    """
    # 1. Add k random transaction tokens
    random_transactions = torch.randint(0, 409, (k,))
    adversarial_sequence = torch.cat((sequence, random_transactions), dim=0)

    # 2. Get BERT's vocabulary probabilities for added tokens
    added_token_indices = adversarial_sequence[-k:]
    vocabulary_probabilities = []
    for token_index in added_token_indices:
        #probabilities = bert_model.predict_token_probabilities(adversarial_sequence, token_index, tokenizer) # Implement this
        probabilities = torch.rand(409) # Placeholder
        vocabulary_probabilities.append(probabilities)

    # 3. Sample from the categorical distribution
    sampled_tokens = []
    for probs in vocabulary_probabilities:
        sampled_token = torch.multinomial(F.softmax(probs / temperature, dim=-1), 1) # Sample based on probabilities
        sampled_tokens.append(sampled_token)

    # 4. Create the adversarial sequence with sampled tokens
    best_adversarial_sequence = adversarial_sequence.clone()
    for i, sampled_token in enumerate(sampled_tokens):
        best_adversarial_sequence[-k + i] = sampled_token

    return best_adversarial_sequence

def seq_concat_sampling_fool_attack(model, bert_model, tokenizer, sequence, target_label, k=2):
    """
    Generates adversarial examples by concatenating tokens sequentially and sampling with BERT.

    Args:
        model: The target model.
        bert_model: A pre-trained BERT model.
        tokenizer: Tokenizer for BERT.
        sequence: The original sequence.
        target_label: The target label.
        k: The number of transactions to add.

    Returns:
        adversarial_sequence: The generated adversarial sequence.
    """
    adversarial_sequence = sequence.clone()
    for _ in range(k):
        # 1. Add a random transaction token
        random_transaction = torch.randint(0, 409, (1,))
        adversarial_sequence = torch.cat((adversarial_sequence, random_transaction), dim=0)

        # 2. Get BERT's vocabulary probabilities for the last added token
        last_token_index = len(adversarial_sequence) - 1
        #probabilities = bert_model.predict_token_probabilities(adversarial_sequence, last_token_index, tokenizer)
        probabilities = torch.rand(409)

        # 3. Sample a token that minimizes the target class probability
        sampled_token = sample_token_minimize_target_prob(model, adversarial_sequence, probabilities, target_label)  # Implement this

        # 4. Update the adversarial sequence
        adversarial_sequence[-1] = sampled_token

    return adversarial_sequence

# Evaluation Metrics

def word_error_rate(original_sequence, adversarial_sequence):
    """
    Calculates the Word Error Rate (WER) between two sequences.

    Args:
        original_sequence (torch.Tensor): The original sequence.
        adversarial_sequence (torch.Tensor): The adversarial sequence.

    Returns:
        float: The Word Error Rate.
    """
    if len(original_sequence) == 0:
        return 0.0 if len(adversarial_sequence) == 0 else 1.0

    edits = 0
    i = 0
    j = 0

    while i < len(original_sequence) and j < len(adversarial_sequence):
        if original_sequence[i] != adversarial_sequence[j]:
            edits += 1
        i += 1
        j += 1

    edits += abs(len(original_sequence) - len(adversarial_sequence))
    return float(edits) / max(len(original_sequence), len(adversarial_sequence))

def adversarial_accuracy(model, original_sequences, adversarial_sequences):
    """
    Calculates the Adversarial Accuracy.

    Args:
        model: The target model.
        original_sequences (torch.Tensor): The original sequences.
        adversarial_sequences (torch.Tensor): The adversarial sequences.

    Returns:
        float: The Adversarial Accuracy.
    """
    correct_classifications = 0
    for i in range(len(original_sequences)):
        original_output = model(original_sequences[i].unsqueeze(0))
        _, original_predicted = torch.max(original_output.data, 1)

        adversarial_output = model(adversarial_sequences[i].unsqueeze(0))
        _, adversarial_predicted = torch.max(adversarial_output.data, 1)

        if original_predicted == adversarial_predicted:
            correct_classifications += 1

    return correct_classifications / len(original_sequences)

def probability_difference(model, original_sequences, adversarial_sequences):
    """
    Calculates the Probability Difference (PD).

    Args:
        model: The target model.
        original_sequences (torch.Tensor): The original sequences.
        adversarial_sequences (torch.Tensor): The adversarial sequences.

    Returns:
        float: The Probability Difference.
    """
    total_pd = 0.0
    for i in range(len(original_sequences)):
        original_output = model(original_sequences[i].unsqueeze(0))
        original_probabilities = torch.softmax(original_output, dim=-1)
        original_max_prob = torch.max(original_probabilities, dim=1)[0].item()

        adversarial_output = model(adversarial_sequences[i].unsqueeze(0))
        adversarial_probabilities = torch.softmax(adversarial_output, dim=-1)
        adversarial_max_prob = torch.max(adversarial_probabilities, dim=1)[0].item()

        total_pd += original_max_prob - adversarial_max_prob

    return total_pd / len(original_sequences)

def normalized_accuracy_drop(model, original_sequences, adversarial_sequences):
    """
    Calculates the Normalized Accuracy Drop (NAD).

    Args:
        model: The target model.
        original_sequences (torch.Tensor): The original sequences.
        adversarial_sequences (torch.Tensor): The adversarial sequences.

    Returns:
        float: The Normalized Accuracy Drop.
    """
    nad = 0.0
    for i in range(len(original_sequences)):
      original_output = model(original_sequences[i].unsqueeze(0))
      _, original_predicted = torch.max(original_output.data, 1)

      adversarial_output = model(adversarial_sequences[i].unsqueeze(0))
      _, adversarial_predicted = torch.max(adversarial_output.data, 1)

      if original_predicted != adversarial_predicted:
        nad +=1
    
    total_wer = 0.0
    for i in range(len(original_sequences)):
        total_wer += word_error_rate(original_sequences[i], adversarial_sequences[i])
    
    return (nad / len(original_sequences))/ (total_wer / len(original_sequences))

if __name__ == '__main__':
    # Example usage:
    dataset_name = "Age 1"
    model_type = "GRU"  # You can change this

    target_model, substitute_model, num_classes = get_model_and_target(dataset_name, model_type)

    print(f"Dataset: {dataset_name}")
    print(f"Model Type: {model_type}")
    print("Target Model:", target_model)
    print("Substitute Model:", substitute_model)
    print("Number of Classes:", num_classes)

    # Example data (replace with your actual data loading)
    # Example: A batch of 10 sequences, each with a maximum length of 20
    batch_size = 10
    max_length = 20
    original_sequences = [torch.randint(0, 409, (random.randint(10, max_length),)) for _ in range(batch_size)] # List of tensors

    # Example target label (replace with your actual target labels)
    target_label = torch.randint(0, num_classes, (batch_size,))

    # Initialize tokenizer and MLM (replace with your actual models)
    tokenizer = None
    mlm = None
    bert_model = None

    # Perform attacks
    adversarial_sequences_sf = [sampling_fool_attack(target_model, tokenizer, mlm, seq) for i, seq in enumerate(original_sequences)]
    # adversarial_sequences_fgsm = [fgsm_attack(target_model, seq, target_label=target_label[i]) for i, seq in enumerate(original_sequences)]
    # adversarial_sequences_concat_fgsm = [concat_fgsm_attack(target_model, seq, target_label=target_label[i]) for i, seq in enumerate(original_sequences)]
    # adversarial_sequences_lm_fgsm = [lm_fgsm_attack(target_model, None, None, seq, target_label=target_label[i], perplexity_threshold=10) for i, seq in enumerate(original_sequences)]
    adversarial_sequences_concat_sf = [concat_sampling_fool_attack(target_model, bert_model, tokenizer, seq, target_label=target_label[i]) for i, seq in enumerate(original_sequences)]
    adversarial_sequences_seq_concat_sf = [seq_concat_sampling_fool_attack(target_model, bert_model, tokenizer, seq, target_label=target_label[i]) for i, seq in enumerate(original_sequences)]

    # Evaluate attacks
    print("\nEvaluation Metrics:")
    print("Attack", "\t", "WER", "\t", "AA", "\t", "PD", "\t", "NAD")

    # Pad sequences to have the same length for metric calculation
    padded_original_sequences = torch.nn.utils.rnn.pad_sequence(original_sequences, batch_first=True)
    padded_adversarial_sequences_sf = torch.nn.utils.rnn.pad_sequence(adversarial_sequences_sf, batch_first=True)
    # padded_adversarial_sequences_fgsm = torch.nn.utils.rnn.pad_sequence(adversarial_sequences_fgsm, batch_first=True)
    # padded_adversarial_sequences_concat_fgsm = torch.nn.utils.rnn.pad_sequence(adversarial_sequences_concat_fgsm, batch_first=True)
    # padded_adversarial_sequences_lm_fgsm = torch.nn.utils.rnn.pad_sequence(adversarial_sequences_lm_fgsm, batch_first=True)
    padded_adversarial_sequences_concat_sf = torch.nn.utils.rnn.pad_sequence(adversarial_sequences_concat_sf, batch_first=True)
    padded_adversarial_sequences_seq_concat_sf = torch.nn.utils.rnn.pad_sequence(adversarial_sequences_seq_concat_sf, batch_first=True)

    #WER
    wer_sf = sum([word_error_rate(original_sequences[i], adversarial_sequences_sf[i]) for i in range(len(original_sequences))]) / len(original_sequences)
    # wer_fgsm = sum([word_error_rate(original_sequences[i], adversarial_sequences_fgsm[i]) for i in range(len(original_sequences))]) / len(original_sequences)
    # wer_concat_fgsm = sum([word_error_rate(original_sequences[i], adversarial_sequences_concat_fgsm[i]) for i in range(len(original_sequences))]) / len(original_sequences)
    # wer_lm_fgsm = sum([word_error_rate(original_sequences[i], adversarial_sequences_lm_fgsm[i]) for i in range(len(original_sequences))]) / len(original_sequences)
    wer_concat_sf = sum([word_error_rate(original_sequences[i], adversarial_sequences_concat_sf[i]) for i in range(len(original_sequences))]) / len(original_sequences)
    wer_seq_concat_sf = sum([word_error_rate(original_sequences[i], adversarial_sequences_seq_concat_sf[i]) for i in range(len(original_sequences))]) / len(original_sequences)

    #AA
    aa_sf = adversarial_accuracy(target_model, padded_original_sequences, padded_adversarial_sequences_sf)
    # aa_fgsm = adversarial_accuracy(target_model, padded_original_sequences, padded_adversarial_sequences_fgsm)
    # aa_concat_fgsm = adversarial_accuracy(target_model, padded_original_sequences, padded_adversarial_sequences_concat_fgsm)
    # aa_lm_fgsm = adversarial_accuracy(target_model, padded_original_sequences, padded_adversarial_sequences_lm_fgsm)
    aa_concat_sf = adversarial_accuracy(target_model, padded_original_sequences, padded_adversarial_sequences_concat_sf)
    aa_seq_concat_sf = adversarial_accuracy(target_model, padded_original_sequences, padded_adversarial_sequences_seq_concat_sf)

    #PD
    pd_sf = probability_difference(target_model, padded_original_sequences, padded_adversarial_sequences_sf)
    # pd_fgsm = probability_difference(target_model, padded_original_sequences, padded_adversarial_sequences_fgsm)
    # pd_concat_fgsm = probability_difference(target_model, padded_original_sequences, padded_adversarial_sequences_concat_fgsm)
    # pd_lm_fgsm = probability_difference(target_model, padded_original_sequences, padded_adversarial_sequences_lm_fgsm)
    pd_concat_sf = probability_difference(target_model, padded_original_sequences, padded_adversarial_sequences_concat_sf)
    pd_seq_concat_sf = probability_difference(target_model, padded_original_sequences, padded_adversarial_sequences_seq_concat_sf)

    #NAD
    nad_sf = normalized_accuracy_drop(target_model, padded_original_sequences, padded_adversarial_sequences_sf)
    # nad_fgsm = normalized_accuracy_drop(target_model, padded_original_sequences, padded_adversarial_sequences_fgsm)
    # nad_concat_fgsm = normalized_accuracy_drop(target_model, padded_original_sequences, padded_adversarial_sequences_concat_fgsm)
    # nad_lm_fgsm = normalized_accuracy_drop(target_model, padded_original_sequences, padded_adversarial_sequences_lm_fgsm)
    nad_concat_sf = normalized_accuracy_drop(target_model, padded_original_sequences, padded_adversarial_sequences_concat_sf)
    nad_seq_concat_sf = normalized_accuracy_drop(target_model, padded_original_sequences, padded_adversarial_sequences_seq_concat_sf)

    # Print results
    print("\nSampling Fool: ")
    print(f"  WER: {wer_sf:.4f}, AA: {aa_sf:.4f}, PD: {pd_sf:.4f}, NAD: {nad_sf:.4f}")

    # print("\nFGSM: ")
    # print(f"  WER: {wer_fgsm:.4f}, AA: {aa_fgsm:.4f}, PD: {pd_fgsm:.4f}, NAD: {nad_fgsm:.4f}")

    # print("\nConcat FGSM: ")
    # print(f"  WER: {wer_concat_fgsm:.4f}, AA: {aa_concat_fgsm:.4f}, PD: {pd_concat_fgsm:.4f}, NAD: {nad_concat_fgsm:.4f}")

    # print("\nLM FGSM: ")
    # print(f"  WER: {wer_lm_fgsm:.4f}, AA: {aa_lm_fgsm:.4f}, PD: {pd_lm_fgsm:.4f}, NAD: {nad_lm_fgsm:.4f}")

    print("\nConcat SF: ")
    print(f"  WER: {wer_concat_sf:.4f}, AA: {aa_concat_sf:.4f}, PD: {pd_concat_sf:.4f}, NAD: {nad_concat_sf:.4f}")

    print("\nSeq Concat SF: ")
    print(f"  WER: {wer_seq_concat_sf:.4f}, AA: {aa_seq_concat_sf:.4f}, PD: {pd_seq_concat_sf:.4f}, NAD: {nad_seq_concat_sf:.4f}")



# Evaluation Metrics:
# Attack   WER     AA      PD      NAD

# Sampling Fool: 
#   WER: 0.2596, AA: 0.8000, PD: 0.0093, NAD: 1.1111

# Concat SF: 
#   WER: 0.1283, AA: 0.8000, PD: 0.0017, NAD: 1.1892

# Seq Concat SF: 
#   WER: 0.1283, AA: 0.7000, PD: 0.0001, NAD: 1.7838
