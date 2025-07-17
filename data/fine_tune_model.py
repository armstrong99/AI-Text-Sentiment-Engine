from transformers import TrainingArguments, Trainer
from datetime import datetime

class CustomModelTrainer:
    def __init__(self, model, tokenizer, tokenized_train_dataset, tokenized_val_dataset  ):
        self.model = model
        self.tokenizer = tokenizer
        self.last_trained = datetime.now()
        self.training_args = TrainingArguments(
                    output_dir="results",
                    num_train_epochs=3,
                    per_device_train_batch_size=8,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    logging_steps=10,
                    logging_dir="logs"
                )
        self.trainer = Trainer(
                    model = model,
                    args=self.training_args,
                    train_dataset=tokenized_train_dataset,
                    eval_dataset=tokenized_val_dataset,
                    tokenizer=tokenizer
                )

    def train_model(self):
        self.trainer.train()
    
    def save_model(self, output_dir="saved_model"):
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
