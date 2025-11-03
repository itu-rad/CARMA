# Import necessary libraries
import os
import datasets
import torch
from transformers import AutoTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from modelsummary import summary
import time

start = time.perf_counter()

# Suppress HF logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Step 1: Load the WikiText dataset
raw_datasets = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"Available splits in the dataset: {raw_datasets.keys()}")

# Step 2: Initialize the tokenizer (fast; no sentencepiece needed)
tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased", use_fast=True)

# Tokenization function
def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, max_length=512)
    result["labels"] = [1] * len(result["input_ids"])  # Dummy label
    return result

# Apply tokenization (drop untouched columns)
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
print(f"Available splits in the tokenized dataset: {tokenized_datasets.keys()}")

# Step 3: Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 4: Training arguments (rename evaluation_strategy -> eval_strategy)
training_args = TrainingArguments(
    output_dir="./xlnet-wiki-output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="no",         # <- new arg name
    save_strategy="no",
    logging_strategy="no",
    save_total_limit=None,
    report_to=[],
    do_train=True,
    do_eval=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 5: Model
model = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased").to(device)

# ==== quick dummy-run + summary (optional) ====
batch_size = 4
seq_length = 512
mock_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), dtype=torch.long, device=device)

try:
    _ = model(input_ids=mock_input)  # be explicit with keyword
    print("Dummy input works with the model.")
except Exception as e:
    print(f"Error during dummy input check: {e}")

try:
    summary(model, mock_input, show_input=True, show_hierarchical=True)
except Exception as e:
    print(f"Error during summary: {e}")
# ==============================================

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 7: Train
trainer.train()

# Step 8: (Optional) Evaluate — you can remove this since do_eval=False
results = trainer.evaluate()
print(f"Evaluation results: {results}")

end = time.perf_counter()
print("\nexecution time: ", end - start)
