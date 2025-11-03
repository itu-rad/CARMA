import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from torch.optim import AdamW

from tqdm import tqdm

from modelsummary import summary  # Import modelsummary

import time

start = time.perf_counter()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a small text dataset from Hugging Face Datasets
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512, return_special_tokens_mask=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Create data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Define a PyTorch DataLoader
train_dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True, collate_fn=data_collator)

# Initialize BERT configuration from scratch
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=1024,             # Hidden size of the embeddings
    num_hidden_layers=24,         # Number of hidden layers (transformer blocks)
    num_attention_heads=16,       # Number of attention heads in each attention layer
    intermediate_size=4096,       # Size of the "intermediate" (feed-forward) layer
    max_position_embeddings=512,  # Maximum length of input sequences
    type_vocab_size=2,            # Number of token types (e.g., token type embeddings for sentence pair classification)
    hidden_act="gelu",            # Activation function used in the feed-forward layers
    initializer_range=0.02,       # Standard deviation of the truncated normal initializer for weights
    layer_norm_eps=1e-12,          # Epsilon used by layer normalization
    dropout=0.1                    # Dropout rate applied to all fully connected layers
)

# Initialize BERT model for Masked Language Modeling
model = BertForMaskedLM(config=config).to(device)





#====
# Create a mock input tensor for the model
batch_size = 8      # Define your batch size
seq_length = 512    # Define sequence length (up to `max_position_embeddings`)
mock_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.long).to(device)

# Check model compatibility with the mock input
try:
    model(mock_input)  # Test with the mock input
    print("Dummy input works with the model.")
except Exception as e:
    print(f"Error during dummy input check: {e}")

# Generate model summary
try:
    summary(model, mock_input, show_input=True, show_hierarchical=True)
except Exception as e:
    print(f"Error during summary: {e}")
#====






# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
epochs = 1
for epoch in range(epochs):
    print("\nepoch ", epoch+1, " started ...")
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # if step % 100 == 0:
        #     print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss}")

# Save the model
# model.save_pretrained("./bert-from-scratch")
# tokenizer.save_pretrained("./bert-from-scratch")

# print("Training completed and model saved.")

end = time.perf_counter()


execution_time = end - start


print("\nexecution time: ", execution_time)