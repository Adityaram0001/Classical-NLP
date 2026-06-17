import pandas as pd
from datasets import load_dataset
from pathlib import Path

def download_and_sample():
    print("Downloading Quora Question Pairs from HuggingFace (SetFit/qqp)...")
    dataset = load_dataset("SetFit/qqp", split="train")
    
    print(f"Total pairs available: {len(dataset)}")
    
    # We sample 100,000 pairs to make pairwise distance computation locally viable
    print("Sampling 100,000 question pairs for feature engineering...")
    sampled_dataset = dataset.shuffle(seed=42).select(range(100000))
    
    df = sampled_dataset.to_pandas()
    
    # Select only the relevant columns for clarity
    df = df[['text1', 'text2', 'label']]
    df.columns = ['question1', 'question2', 'is_duplicate']
    
    # Drop rows with nulls
    df = df.dropna()
    
    out_dir = Path("datasets/05_quora_question_pairs/data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sampled dataset with {len(df)} rows to {out_path}")

if __name__ == "__main__":
    download_and_sample()
