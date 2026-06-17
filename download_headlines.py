import pandas as pd
from datasets import load_dataset
from pathlib import Path

def download_and_sample():
    print("Downloading A Million News Headlines from HuggingFace...")
    # The dataset 'rajistics/million-headlines' has a single 'train' split
    dataset = load_dataset("rajistics/million-headlines", split="train")
    
    print(f"Total headlines available: {len(dataset)}")
    
    print("Sampling 100,000 headlines for efficient local processing...")
    # Seed set for reproducibility
    sampled_dataset = dataset.shuffle(seed=42).select(range(100000))
    
    df = sampled_dataset.to_pandas()
    
    out_dir = Path("datasets/04_topic_modeling_news/data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sampled dataset with {len(df)} rows to {out_path}")

if __name__ == "__main__":
    download_and_sample()
