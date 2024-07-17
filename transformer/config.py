def get_config() -> dict:
    return {
        "device": "cpu",
        "seed": 50,
        "batch_size": 100,
        "num_epochs": 20,
        "lr": 10 ** -4,
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
