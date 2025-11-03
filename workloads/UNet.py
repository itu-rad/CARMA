import os, argparse, torch, torch.utils.data as data
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image


from tqdm.auto import tqdm
from torchinfo import summary
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

class VOCSeg(data.Dataset):
    def __init__(self, root, year="2012", image_set="train", img_size=512):
        self.base = VOCSegmentation(root=root, year=year, image_set=image_set, download=False)
        self.img_size = img_size
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        # VOC masks are 21 classes [0..20], 255 = ignore
        self.mask_tf = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)])

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        img, mask = self.base[i]
        img = self.img_tf(img)
        mask = np.array(self.mask_tf(mask), dtype=np.int64)
        mask[mask == 255] = 0
        return img, torch.from_numpy(mask)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/raid/datasets")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    print(f"AMP enabled: {args.amp}")

    train = VOCSeg(args.root, image_set="train", img_size=args.size)
    val   = VOCSeg(args.root, image_set="val",   img_size=args.size)
    train_loader = data.DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = data.DataLoader(val,   batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda")

    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=21).to(device)

    faketensor_time1 = time.perf_counter()
    print(fn(model, args.bs, (3, args.size, args.size)))
    faketensor_time2 = time.perf_counter()

    faketensor_time = faketensor_time2 - faketensor_time1
    print("Time taken by faketensor: ", faketensor_time)


    summary(
    model,
    input_size=(args.bs, 3, args.size, args.size),  # batch-first here
    device=str(device),
    col_names=("kernel_size","input_size","output_size","num_params","mult_adds"),
    depth=3,
    )
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss = smp.losses.DiceLoss(mode="multiclass")
    scaler = torch.amp.GradScaler(enabled=args.amp)

    def run_epoch(loader, train=True):
        model.train(train)
        total, n = 0.0, 0
        # NEW: wrap loader with tqdm progress bar
        pbar = tqdm(loader, total=len(loader), desc="train" if train else "val", leave=False)
        for x, y in pbar:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                logits = model(x)
                L = loss(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(L).backward()
                scaler.step(opt)
                scaler.update()
            total += L.item() * x.size(0); n += x.size(0)
            # NEW: optional live display of running loss on the bar
            pbar.set_postfix({'loss': f"{(total/max(1,n)):.4f}"})
        return total / max(1, n)
    
    start = time.perf_counter()

    for e in range(args.epochs):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        print(f"epoch {e+1}: train_dice_loss={tr:.4f} val_dice_loss={va:.4f}")

    end = time.perf_counter()
    execution_time = end - start
    print("\n execution time: ", execution_time)

if __name__ == "__main__":
    main()