import os
import shutil
from allennlp.commands.train import train_model_from_file

# Create main directories
os.makedirs("experiments/trained_models/", exist_ok=True)

datasets = ["age", "age_short", "age_tinkoff", "rosbank"]
clf_types = ["target", "substitute"]
config_names = ["gru_with_amounts", "lstm_with_amounts", "cnn_with_amounts"]

discretizer_name = "100_quantile"
random_seed = "0"

for dataset_name in datasets:
    dataset_path = f"experiments/trained_models/{dataset_name}/"
    os.makedirs(dataset_path, exist_ok=True)

    for clf_type in clf_types:
        clf_path = os.path.join(dataset_path, f"{clf_type}_clf")
        os.makedirs(clf_path, exist_ok=True)

        for config_name in config_names:
            model_dir = f"experiments/trained_models/{dataset_name}/{clf_type}_clf/{config_name}"

            # Remove existing model directory
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            # Set variable paths
            clf_train_data_path = f"data/{dataset_name}/{clf_type}_clf/train.jsonl"
            clf_valid_data_path = f"data/{dataset_name}/{clf_type}_clf/valid.jsonl"
            clf_test_data_path = f"data/{dataset_name}/test.jsonl"
            discretizer_path = f"presets/{dataset_name}/discretizers/{discretizer_name}"
            vocab_path = f"presets/{dataset_name}/vocabs/{discretizer_name}"

            # Set overrides for Jsonnet config
            overrides = {
                "CLF_TRAIN_DATA_PATH": clf_train_data_path,
                "CLF_VALID_DATA_PATH": clf_valid_data_path,
                "CLF_TEST_DATA_PATH": clf_test_data_path,
                "DISCRETIZER_PATH": discretizer_path,
                "VOCAB_PATH": vocab_path,
                "RANDOM_SEED": random_seed
            }

            # Run AllenNLP train from Python
            params_file = f"configs/classifiers/{config_name}.jsonnet"
            train_model_from_file(
                parameter_filename=params_file,
                serialization_dir=model_dir,
                include_package="advsber",
                overrides=overrides
            )

            # Define file paths
            model_src_path = f"experiments/trained_models/{dataset_name}/{clf_type}_clf/{config_name}/model.tar.gz"
            model_dst_dir = f"presets/{dataset_name}/models/{clf_type}_clf/"
            model_dst_path = os.path.join(model_dst_dir, f"{config_name}.tar.gz")

            # Ensure destination directory exists
            os.makedirs(model_dst_dir, exist_ok=True)

            # Remove existing tar.gz file if it exists
            if os.path.exists(model_dst_path):
                os.remove(model_dst_path)

            # Copy model file
            shutil.copy(model_src_path, model_dst_path)
