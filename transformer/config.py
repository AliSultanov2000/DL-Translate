from pathlib import Path

def get_config() -> dict:
    return {
        "device": "cpu",
        "seed": 17,
        "batch_size": 15,
        "num_epochs": 20,
        "lr": 10**-4,
        "num_examples": 2,
        "console_width": 130,
        "max_len": 250, 
        "seq_len": 250,
        "d_model": 512,
        "temperature": 0.8,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "ru",
        "model_folder": "transformer_weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config: dict, epoch: str) -> str:
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config: dict) -> str:
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])