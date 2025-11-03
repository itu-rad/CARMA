import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from torchsummary import summary
import argparse
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

def tensor_storage_id(tensor):
    return tensor._typed_storage()._cdata

class FakeTensorMemoryProfilerMode(TorchDispatchMode):
    def __init__(self):
        # counter of storage ids to live references
        self.storage_count: Dict[int, int] = Counter()
        # live fake tensors
        self.live_tensors = WeakIdKeyDictionary()
        self.memory_use = 0
        self.max_memory = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        rs = func(*args, **kwargs)
        tree_map_only(torch._subclasses.FakeTensor, self.track_tensor_memory_use, rs)
        return rs

    def track_tensor_memory_use(self, tensor):
        # already accounted for
        if tensor in self.live_tensors:
            return

        self.live_tensors[tensor] = True
        nbytes = tensor.untyped_storage().nbytes()

        storage_id = tensor_storage_id(tensor)

        # new storage, add to memory
        if storage_id not in self.storage_count:
            self.change_memory(nbytes)

        self.storage_count[storage_id] += 1

        # when this tensor dies, we need to adjust memory
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

def fn(model, batch_size, d):
    print("got it: ", d[0])
    print(f"Running batch size {batch_size}")
    with FakeTensorMode(allow_non_fake_inputs=True):
        with FakeTensorMemoryProfilerMode() as ftmp:
            device = 'cuda'
            fake_input = torch.rand([batch_size, d[0], d[1], d[2]], requires_grad=True).to(device)
            model = model.to(device)
            output = model(fake_input)
            # output = model(, requires_grad=True)).to('cuda')
            print(f"GB after forward: {ftmp.max_memory / GB}")
            output.sum().backward()
            print(f"GB after backward: {ftmp.max_memory / GB}")
            return ftmp.max_memory 
# added by ehsan
var = ()
# added by Ehsan for time control
start_time =0
max_duration = 3 * 60
# ===============================


start = time.perf_counter()

# Command-line arguments
parser = argparse.ArgumentParser(description='Train EfficientNet-B0 on CIFAR-10')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
args = parser.parse_args()

# Check for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model creation
model = create_model('efficientnet_b0', pretrained=False, num_classes=100)
model = model.to(device)


faketensor_time1 = time.perf_counter()
print(fn(model, int(args.batch_size), (3, 32, 32)))
faketensor_time2 = time.perf_counter()

faketensor_time = faketensor_time2 - faketensor_time1
print("Time taken by faketensor: ", faketensor_time)


# Print model summary
print("Model Summary:")
summary(model, input_size=(3, 32, 32))

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%")

# Test function
def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f} | Test Acc: {100. * correct / total:.2f}%")
    return 100. * correct / total

# Main loop
best_acc = 0.0
for epoch in range(1, args.epochs + 1):  # Train for specified epochs
    train(epoch)
    acc = test()
    scheduler.step()

    # Save the model if accuracy improves
    # if acc > best_acc:
    #     print(f"Saving model with accuracy: {acc:.2f}%")
    #     torch.save(model.state_dict(), 'efficientnet_b0_cifar10.pth')
    #     best_acc = acc


end = time.perf_counter()

print("time: ", end - start)