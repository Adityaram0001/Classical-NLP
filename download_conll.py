import urllib.request
import zipfile
import json
import os
from pathlib import Path

def download_conll2003():
    print("Downloading CoNLL-2003 zip from data.deepai.org...")
    zip_url = "https://data.deepai.org/conll2003.zip"
    
    out_dir = Path("datasets/06_conll2003_ner/data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "conll2003.zip"
    
    urllib.request.urlretrieve(zip_url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
        
    def process_file(txt_file, split_name):
        print(f"Parsing {split_name}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        sentences = []
        current_tokens = []
        current_pos = []
        current_ner = []
        
        for line in raw_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('-DOCSTART-'):
                if current_tokens:
                    sentences.append({
                        'tokens': current_tokens,
                        'pos_tags': current_pos,
                        'ner_tags': current_ner
                    })
                    current_tokens, current_pos, current_ner = [], [], []
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                current_tokens.append(parts[0])
                current_pos.append(parts[1])
                current_ner.append(parts[3])
                
        if current_tokens:
             sentences.append({
                  'tokens': current_tokens,
                  'pos_tags': current_pos,
                  'ner_tags': current_ner
             })
             
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, 'w') as f:
            for s in sentences:
                f.write(json.dumps(s) + '\n')
        print(f"Saved {len(sentences)} sentences to {out_path}")

    process_file(out_dir / "train.txt", 'train')
    process_file(out_dir / "test.txt", 'test')
    
    # Cleanup
    os.remove(zip_path)

if __name__ == "__main__":
    download_conll2003()
