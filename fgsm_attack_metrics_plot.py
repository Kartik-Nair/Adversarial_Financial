import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer  # Install with: pip install jiwer

# Word Error Rate (WER)
def compute_wer(original_sequences, adversarial_sequences):
    wers = []
    for orig, adv in zip(original_sequences, adversarial_sequences):
        orig_str = ' '.join(map(str, orig.flatten()))
        adv_str = ' '.join(map(str, adv.flatten()))
        wers.append(wer(orig_str, adv_str))
    return np.mean(wers)

# Adversarial Accuracy (AA)
def compute_adversarial_accuracy(model, X_adv, y_true):
    y_pred = np.argmax(model.predict(X_adv), axis=1)
    return np.mean(y_pred == y_true)

# Probability Difference (PD)
def compute_probability_difference(model, X_orig, X_adv, y_true):
    orig_probs = model.predict(X_orig)
    adv_probs = model.predict(X_adv)
    orig_conf = np.array([orig_probs[i, label] for i, label in enumerate(y_true)])
    adv_conf = np.array([adv_probs[i, label] for i, label in enumerate(y_true)])
    return np.mean(orig_conf - adv_conf)

# Normalized Accuracy Drop (NAD)
def compute_nad(model, X_orig, X_adv, y_true):
    y_orig_pred = np.argmax(model.predict(X_orig), axis=1)
    y_adv_pred = np.argmax(model.predict(X_adv), axis=1)
    incorrect_flags = (y_orig_pred != y_adv_pred).astype(int)

    wers = []
    for orig, adv in zip(X_orig, X_adv):
        orig_str = ' '.join(map(str, orig.flatten()))
        adv_str = ' '.join(map(str, adv.flatten()))
        wers.append(wer(orig_str, adv_str))

    nad_values = [inc / (w + 1e-6) for inc, w in zip(incorrect_flags, wers)]
    return np.mean(nad_values)

# FGSM Attack function (to test with different epsilon and steps)
def fgsm_attack(model, X, y, epsilon=0.1, num_steps=30):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor)
            loss = sparse_categorical_crossentropy(y_tensor, predictions)
        
        gradients = tape.gradient(loss, X_tensor)
        perturbations = tf.sign(gradients)
        X_tensor = X_tensor + epsilon * perturbations
        X_tensor = tf.clip_by_value(X_tensor, 0.0, 1.0)
    
    return X_tensor.numpy()

# === Usage ===
# Assuming you already have: model, X_test_tensor, X_adv, y_test_tensor

# Define the values for epsilon and number of steps to test
epsilons = [0.1, 0.2]
steps = [5, 10]

# Store the metrics results for different epsilon and steps
aa_values = []
pd_values = []
wer_values = []
nad_values = []

# Iterate over epsilon and steps to track the metrics
for epsilon in epsilons:
    for num_steps in steps:
        # Perform FGSM attack
        X_adv = fgsm_attack(model, X_test_tensor, y_test_tensor, epsilon, num_steps)

        # Compute metrics
        aa = compute_adversarial_accuracy(model, X_adv, y_test_tensor)
        pd = compute_probability_difference(model, X_test_tensor, X_adv, y_test_tensor)
        wer_score = compute_wer(X_test_tensor, X_adv)
        nad = compute_nad(model, X_test_tensor, X_adv, y_test_tensor)

        # Store the results for plotting
        aa_values.append(aa)
        pd_values.append(pd)
        wer_values.append(wer_score)
        nad_values.append(nad)

# Reshape the results for plotting (each epsilon is tested with different steps)
aa_values = np.array(aa_values).reshape(len(epsilons), len(steps))
pd_values = np.array(pd_values).reshape(len(epsilons), len(steps))
wer_values = np.array(wer_values).reshape(len(epsilons), len(steps))
nad_values = np.array(nad_values).reshape(len(epsilons), len(steps))

# === Visualization ===
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Plot Adversarial Accuracy (AA)
cax = axs[0, 0].imshow(aa_values, aspect='auto', cmap='viridis', origin='lower')
axs[0, 0].set_title("Adversarial Accuracy (AA)")
axs[0, 0].set_xticks(np.arange(len(steps)))
axs[0, 0].set_xticklabels(steps)
axs[0, 0].set_yticks(np.arange(len(epsilons)))
axs[0, 0].set_yticklabels(epsilons)
axs[0, 0].set_xlabel("Number of Steps")
axs[0, 0].set_ylabel("Epsilon")
fig.colorbar(cax, ax=axs[0, 0])

# Plot Probability Difference (PD)
cax = axs[0, 1].imshow(pd_values, aspect='auto', cmap='viridis', origin='lower')
axs[0, 1].set_title("Probability Difference (PD)")
axs[0, 1].set_xticks(np.arange(len(steps)))
axs[0, 1].set_xticklabels(steps)
axs[0, 1].set_yticks(np.arange(len(epsilons)))
axs[0, 1].set_yticklabels(epsilons)
axs[0, 1].set_xlabel("Number of Steps")
axs[0, 1].set_ylabel("Epsilon")
fig.colorbar(cax, ax=axs[0, 1])

# Plot Word Error Rate (WER)
cax = axs[1, 0].imshow(wer_values, aspect='auto', cmap='viridis', origin='lower')
axs[1, 0].set_title("Word Error Rate (WER)")
axs[1, 0].set_xticks(np.arange(len(steps)))
axs[1, 0].set_xticklabels(steps)
axs[1, 0].set_yticks(np.arange(len(epsilons)))
axs[1, 0].set_yticklabels(epsilons)
axs[1, 0].set_xlabel("Number of Steps")
axs[1, 0].set_ylabel("Epsilon")
fig.colorbar(cax, ax=axs[1, 0])

# Plot Normalized Accuracy Drop (NAD)
cax = axs[1, 1].imshow(nad_values, aspect='auto', cmap='viridis', origin='lower')
axs[1, 1].set_title("Normalized Accuracy Drop (NAD)")
axs[1, 1].set_xticks(np.arange(len(steps)))
axs[1, 1].set_xticklabels(steps)
axs[1, 1].set_yticks(np.arange(len(epsilons)))
axs[1, 1].set_yticklabels(epsilons)
axs[1, 1].set_xlabel("Number of Steps")
axs[1, 1].set_ylabel("Epsilon")
fig.colorbar(cax, ax=axs[1, 1])

# Display the plots
plt.tight_layout()
plt.show()
