import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from torch.optim import AdamW

from tqdm import tqdm

from modelsummary import summary  # Import modelsummary

import time

# added by Ehsan for using tensorfake for memory estimation
from collections import Counter
import functools
import weakref
from typing import Dict

import torch
from torch._subclasses import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.weak import WeakIdKeyDictionary
import torchvision.models as models


# ===================== FakeTensor helpers (UNCHANGED structure) =====================
def tensor_storage_id(tensor):
    return tensor._typed_storage()._cdata

class FakeTensorMemoryProfilerMode(TorchDispatchMode):
    def __init__(self):
        self.storage_count: Dict[int, int] = Counter()
        self.live_tensors = WeakIdKeyDictionary()
        self.memory_use = 0
        self.max_memory = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        rs = func(*args, **kwargs)
        tree_map_only(torch._subclasses.FakeTensor, self.track_tensor_memory_use, rs)
        return rs

    def track_tensor_memory_use(self, tensor):
        if tensor in self.live_tensors:
            return
        self.live_tensors[tensor] = True
        nbytes = tensor.untyped_storage().nbytes()
        storage_id = tensor_storage_id(tensor)
        if storage_id not in self.storage_count:
            self.change_memory(nbytes)
        self.storage_count[storage_id] += 1
        weakref.finalize(tensor, functools.partial(self.tensor_cleanup, storage_id, nbytes))

    def tensor_cleanup(self, storage_id, nbytes):
        self.storage_count[storage_id] -= 1
        if self.storage_count[storage_id] == 0:
            del self.storage_count[storage_id]
            self.change_memory(-nbytes)

    def change_memory(self, delta):
        self.memory_use += delta
        self.max_memory = max(self.memory_use, self.max_memory)


MB = 2 ** 20
GB = 2 ** 30
MEMORY_LIMIT = 40 * GB
# ================================================================================


# >>> CHANGED <<< keep inputs on CPU; FakeTensor will wrap them (no .to(device) here)
def generate_random_input(batch_size, seq_length, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    return input_ids, attention_mask


def fn(model, batch_size, seq_length, vocab_size):
    # 1) Make sure we avoid SDPA/flash kernels & mask utils
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    # HuggingFace switch: force the non-SDPA ("eager") attention implementation
    if hasattr(model, "config"):
        try:
            model.config._attn_implementation = "eager"  # key bit
        except Exception:
            pass

    print(f"[FakeTensor] Estimating for batch={batch_size}, seq={seq_length}")

    # 2) Create inputs on meta so devices match model weights (also meta)
    meta_input_ids = torch.empty(batch_size, seq_length, dtype=torch.long, device="meta")

    # 3) Wrap as fake and run a forward WITHOUT attention_mask (avoids data-dependent torch.all)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake_input_ids = mode.from_tensor(meta_input_ids)
        with FakeTensorMemoryProfilerMode() as ftmp:
            with torch.no_grad():
                _ = model(input_ids=fake_input_ids)  # no attention_mask passed on purpose
            peak_bytes = ftmp.max_memory
            print(f"[FakeTensor] Peak forward (bytes): {peak_bytes}")
            print(f"[FakeTensor] Peak forward (GB): {peak_bytes / (1024**3):.3f}")
            return peak_bytes




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
train_dataloader = DataLoader(tokenized_datasets, batch_size=32, shuffle=True, collate_fn=data_collator)

# Initialize BERT configuration from scratch (UNCHANGED)
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,              # Hidden size of the embeddings
    num_hidden_layers=12,         # Number of hidden layers (transformer blocks)
    num_attention_heads=12,       # Number of attention heads in each attention layer
    intermediate_size=3072,       # Size of the "intermediate" (feed-forward) layer
    max_position_embeddings=512,  # Maximum length of input sequences
    type_vocab_size=2,            # Number of token types
    hidden_act="gelu",            # Activation function
    initializer_range=0.02,       # Init stddev
    layer_norm_eps=1e-12,         # LayerNorm epsilon
    dropout=0.1 
)

# >>> ADDED <<< meta model for FakeTensor estimation (does not allocate real weights)
model_meta = BertForMaskedLM(config=config).to_empty(device="meta").eval()

# >>> ADDED <<< run FakeTensor estimate BEFORE real training model is built
_ = fn(model_meta, batch_size=32, seq_length=512, vocab_size=config.vocab_size)

# ======================= REAL MODEL FOR TRAINING (your original flow) =======================
# Initialize BERT model for Masked Language Modeling (real allocations)
model = BertForMaskedLM(config=config).to(device)

#====
# Create a mock input tensor for the model
batch_size = 32  # Define your batch size
seq_length = 512  # Define sequence length
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

# Training loop (UNCHANGED)
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

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss}")

end = time.perf_counter()
execution_time = end - start
print("\nexecution time: ", execution_time)
