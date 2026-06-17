import pandas as pd
from datasets import load_dataset
from pathlib import Path

def download_and_save():
    print("Downloading IMDB dataset from HuggingFace...")
    dataset = load_dataset("stanfordnlp/imdb")
    
    print("Converting to pandas DataFrames...")
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # Hugging Face IMDB dataset has an 'unsupervised' split, but we only need labeled train/test.
    # The labels are 0 for negative, 1 for positive.
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    out_dir = Path("datasets/03_imdb_movie_reviews/data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset with {len(df)} rows to {out_path}")

if __name__ == "__main__":
    download_and_save()
