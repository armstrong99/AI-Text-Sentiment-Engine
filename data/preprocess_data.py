from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
from config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=256)

def load_and_tokenize_dataset(csv_path: str, val_size=0.2):
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)

#split into train/validation(80% tain & 20% validation)
    split_dataset  = dataset.train_test_split(test_size=val_size,shuffle=True)

    tokenized_train = split_dataset["train"].map(tokenize, batched=True)
    tokenized_val = split_dataset["test"].map(tokenize, batched=True)

# encode labels
    tokenized_train = tokenized_train.class_encode_column("label")
    tokenized_val = tokenized_val.class_encode_column("label")

    return {"train": tokenized_train, "validation": tokenized_val, "tokenizer": tokenizer}
