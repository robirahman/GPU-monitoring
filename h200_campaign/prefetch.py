"""Download every dataset and model the campaign uses, so collection never waits on the network."""
from datasets import load_dataset
from huggingface_hub import snapshot_download

DATASETS = [("uoft-cs/cifar10",), ("nyu-mll/glue", "sst2"), ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            ("tatsu-lab/alpaca",)]
MODELS = ["google-bert/bert-base-uncased", "Qwen/Qwen2.5-0.5B", "microsoft/Phi-3-mini-4k-instruct",
          "openai/whisper-large-v3-turbo"]

if __name__ == "__main__":
    for args in DATASETS:
        ds = load_dataset(*args)
        print("dataset", args, {k: len(v) for k, v in ds.items()}, flush=True)
    ls = load_dataset("openslr/librispeech_asr", "clean", split="validation")
    print("dataset librispeech clean/validation", len(ls), flush=True)
    for m in MODELS:
        p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken",
                                                 "tokenizer*", "*.py"])
        print("model", m, p, flush=True)
    print("PREFETCH_DONE")
