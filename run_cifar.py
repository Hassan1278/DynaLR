# File: examples/run_cifar.py

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler


# 1. Model definitions
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_resnet18(num_classes):
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 2. Utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(dataset='CIFAR10', batch_size=128, val_size=5000):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    if dataset == 'CIFAR10':
        ds = datasets.CIFAR10
    else:
        ds = datasets.CIFAR100

    train_full = ds(root='.', train=True, download=True, transform=transform)
    indices = list(range(len(train_full)))
    random.shuffle(indices)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    train_loader = torch.utils.data.DataLoader(
        train_full, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx), num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        train_full, batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx), num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        ds(root='.', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader, test_loader

# 3. Training & evaluation
def train_one_epoch(model, optimizer, scheduler, loader, device, use_dyn, epoch, name):
    model.train()
    running, correct, total = 0.0, 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if use_dyn:
            optimizer.opt.zero_grad()
        else:
            optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out,y)
        loss.backward()
        if use_dyn:
            optimizer.step(loss)
        else:
            optimizer.step()
            if scheduler:
                scheduler.step()
        running += loss.item()*x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    print(f\"{name} Epoch {epoch}: loss={running/total:.4f}, acc={correct/total:.4f}\")

def evaluate(model, loader, device):
    model.eval()
    loss, correct, total = 0.0,0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss += F.cross_entropy(out,y).item()*x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += x.size(0)
    return loss/total, correct/total

# 4. Benchmark runner
def benchmark(optim_fns, seeds, epochs=30, dataset='CIFAR10', model_fn=SimpleCNN):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data_loaders(dataset)
    results = {}
    for name, fn in optim_fns.items():
        accs, times = [], []
        for seed in seeds:
            set_seed(seed)
            model = model_fn(num_classes=10 if dataset=='CIFAR10' else 100).to(device)
            optimizer, scheduler, use_dyn = fn(model)
            start = time.time()
            for ep in range(1, epochs+1):
                train_one_epoch(model, optimizer, scheduler, train_loader, device, use_dyn, ep, name)
            _, test_acc = evaluate(model, test_loader, device)
            elapsed = time.time() - start
            accs.append(test_acc); times.append(elapsed)
            print(f\"{name} | seed {seed} | acc={test_acc:.4f} | time={elapsed:.1f}s\")
        m,s,t = np.mean(accs), np.std(accs), np.mean(times)
        results[name] = (m,s,t)
        print(f\"--> {name}: {m:.4f}±{s:.4f}, time={t:.1f}s\\n\")
    print(\"Final:\") 
    for name,(m,s,t) in results.items():
        print(f\"{name:>15} | acc={m:.4f}±{s:.4f} | time={t:.1f}s\")

# 5. Optimizer configuration
def get_optimizers():
    import torch.optim as optim
    def make_dyn(cls):
        return lambda m: (cls(optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)), None, True)
    return {
        'AdaptivePID': make_dyn(DynaLRPlusPlusAdaptivePID),
        'NoMemory':    make_dyn(DynaLRPlusPlusNoMemory),
        'Memory':      make_dyn(DynaLRPlusPlusMemory),
        'Enhanced':    make_dyn(DynaLRPlusPlusEnhanced),
        'Adam':        lambda m: (optim.Adam(m.parameters(), lr=1e-3), None, False),
    }

if __name__ == '__main__':
    seeds = [42,123,766]
    optim_fns = get_optimizers()
    benchmark(optim_fns, seeds, epochs=30, dataset='CIFAR10', model_fn=SimpleCNN)\n```
For cifar 100 just change dataset to 'CIFAR100'

```
