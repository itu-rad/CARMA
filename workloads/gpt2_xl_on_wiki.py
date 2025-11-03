import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')

# Add a padding token to the GPT2 tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the padding token


# Tokenize the dataset with truncation
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define a PyTorch DataLoader
train_dataloader = DataLoader(tokenized_datasets, batch_size=2, shuffle=True, collate_fn=data_collator)

# Initialize GPT configuration from scratch
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_ctx=1024,
    n_embd=1600,
    n_layer=48,
    n_head=25,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    scale_attn_weights=True,
    use_cache=True,
    bos_token_id=50256,
    eos_token_id=50256,
    # architecture types are optional and typically used for compatibility checks
    # transformer_model_arch_type="gpt2"
)

# Initialize GPT model for Language Modeling
model = GPT2LMHeadModel(config=config).to(device)


#====
# Create a mock input tensor for the model
batch_size = 2      # Define your batch size
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
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
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
# model.save_pretrained("./gpt-from-scratch")
# tokenizer.save_pretrained("./gpt-from-scratch")

# print("Training completed and model saved.")

end = time.perf_counter()

execution_time = end - start

print("\nexecution time: ", execution_time)