# Import necessary libraries
import os
import datasets
import torch
from transformers import AutoTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from modelsummary import summary
import time

start = time.perf_counter()

# Reduce HF logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 1) Load dataset
raw_datasets = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"Available splits in the dataset: {raw_datasets.keys()}")

# 2) Tokenizer (use fast to avoid sentencepiece requirement)
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased", use_fast=True)

def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, max_length=512)
    # Dummy labels for seq-classification head
    result["labels"] = [1] * len(result["input_ids"])
    return result

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names)
print(f"Available splits in the tokenized dataset: {tokenized_datasets.keys()}")

# 3) Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4) Training args (rename evaluation_strategy -> eval_strategy)
training_args = TrainingArguments(
    output_dir="xlnet_output",
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="no",          # <- new name
    save_strategy="no",
    logging_strategy="no",
    report_to=[],
    do_train=True,
    do_eval=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5) Model
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased").to(device)

# ==== quick dummy-run + summary ====
batch_size = 8
seq_length = 512
mock_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), dtype=torch.long, device=device)

try:
    model(mock_input)  # forward with only input_ids is acceptable
    print("Dummy input works with the model.")
except Exception as e:
    print(f"Error during dummy input check: {e}")

try:
    summary(model, mock_input, show_input=True, show_hierarchical=True)
except Exception as e:
    print(f"Error during summary: {e}")
# ===================================

# 6) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset kept but eval is disabled; you can drop it if you want
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7) Train
trainer.train()

# 8) Evaluate (optional; will run even with do_eval=False because we call it explicitly)
results = trainer.evaluate()
print(f"Evaluation results: {results}")

end = time.perf_counter()
print("\nexecution time: ", end - start)
