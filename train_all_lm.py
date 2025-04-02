import os
import shutil
from allennlp.commands.train import train_model_from_file

# Create main directories (if they don't exist)
os.makedirs("experiments/trained_models/", exist_ok=True)
os.makedirs("presets", exist_ok=True)

datasets = ["age", "age_short", "age_tinkoff", "rosbank"]
config_name = "bert_with_amounts"
discretizer_name = "100_quantile"

for dataset_name in datasets:
    dataset_experiments_path = f"experiments/trained_models/{dataset_name}/"
    lm_experiments_path = os.path.join(dataset_experiments_path, "lm")
    os.makedirs(lm_experiments_path, exist_ok=True)

    model_dir = os.path.join(lm_experiments_path, config_name)

    # Remove existing model directory
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # Set variable paths
    lm_train_data_path = f"data/{dataset_name}/lm/train.jsonl"
    lm_valid_data_path = f"data/{dataset_name}/lm/valid.jsonl"
    # discretizer_path = f"presets/{dataset_name}/discretizers/{discretizer_name}"
    # vocab_path = f"presets/{dataset_name}/vocabs/{discretizer_name}"

    # Set overrides for Jsonnet config
    overrides = {
        "TRAIN_DATA_PATH": lm_train_data_path,
        "VALID_DATA_PATH": lm_valid_data_path,
        # "DISCRETIZER_PATH": discretizer_path,
        # "VOCAB_PATH": vocab_path,
    }

    # Run AllenNLP train from Python
    params_file = f"configs/language_models/{config_name}.jsonnet"
    train_model_from_file(
        parameter_filename=params_file,
        serialization_dir=model_dir,
        include_package="advsber",
        overrides=overrides,
    )

    # Define file paths for copying the trained model
    model_src_path = os.path.join(model_dir, "model.tar.gz")
    model_dst_dir = f"presets/{dataset_name}/models/lm/"
    model_dst_path = os.path.join(model_dst_dir, f"{config_name}.tar.gz")

    # Ensure destination directory exists
    os.makedirs(model_dst_dir, exist_ok=True)

    # Remove existing tar.gz file if it exists
    if os.path.exists(model_dst_path):
        os.remove(model_dst_path)

    # Copy model file
    shutil.copy(model_src_path, model_dst_path)

print("Finished training all language models.")
