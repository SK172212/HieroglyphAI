import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from collections import Counter
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

])

data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

#print("Data dir:", data_dir)
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

full_dataset = datasets.ImageFolder(data_dir)

label_counts = Counter(full_dataset.targets)

#print("Images per class:")
#for class_idx, count in label_counts.items():
#    class_name = full_dataset.classes[class_idx]
#    print(f"{class_name}: {count}")

#print(
#    "Min images per class:", min(label_counts.values()),
#    "Max images per class:", max(label_counts.values())
#)

min_images = 6

kept_class_indices = [
    class_idx
    for class_idx, count in label_counts.items()
    if count >= min_images
]

#print("Number of classes kept:", len(kept_class_indices))

kept_class_indices = sorted(kept_class_indices)
old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_class_indices)}
kept_class_names = [full_dataset.classes[i] for i in kept_class_indices]

num_classes = len(kept_class_names)
#print("Num classes (v1):", num_classes)

#print("Kept classes:", kept_class_names)

filtered_samples = []
filtered_targets = []

for path, old_label in full_dataset.samples:
    if old_label in old_to_new:
        filtered_samples.append((path, old_to_new[old_label]))
        filtered_targets.append(old_to_new[old_label])

#print("Filtered dataset size:", len(filtered_samples))
#print("Filtered num classes:", len(kept_class_names))



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


val_ratio = 0.2
seed = 42
g = torch.Generator().manual_seed(seed)

class_to_indices = {c: [] for c in range(num_classes)}
for i, (_, label) in enumerate(filtered_samples):
    class_to_indices[label].append(i)

train_indices = []
val_indices = []

for c in range (num_classes):
    idxs = class_to_indices[c]
    idxs = torch.tensor(idxs)[torch.randperm(len(idxs), generator=g)].tolist()

    n_val = max(1, int(round(val_ratio * len(idxs))))

    n_val = min(n_val, len(idxs) - 1)

    val_indices.extend(idxs[:n_val])
    train_indices.extend(idxs[n_val:])

train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices), generator=g)].tolist()
val_indices = torch.tensor(val_indices)[torch.randperm(len(val_indices), generator=g)].tolist()

train_samples = [filtered_samples[i] for i in train_indices]
val_samples = [filtered_samples[i] for i in val_indices]

print("Stratified split -> Train:", len(train_samples), "Val:", len(val_samples))

train_dataset = ImageDataset(
    samples = train_samples,
    transform = train_transform

)

val_dataset = ImageDataset(
    samples = val_samples,
    transform = val_transform
)

#print("Train dataset size:", len(train_dataset))
#print("Val dataset size:", len(val_dataset))

batch_size = 16  

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

#print(model.fc)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

num_epochs = 10

best_val_acc = 0.0
save_dir = Path(__file__).resolve().parent.parent / "models"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "resnet18_v1_stratified_10ep_NOflip.pth"

for epoch in range(1, num_epochs+1):
    #train
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_from_logits(logits, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    #validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
        
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch:02d}/{num_epochs} | "
        f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.3f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": kept_class_names,
            "img_size": 224,

        }, save_path)
        print(f" Saved best model to {save_path} (val acc {best_val_acc:.3f})")

ckpt = torch.load(save_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("\nLoaded BEST checkpoint for evaluation:", save_path)

#Confusion Matrix 

model.eval()

C= num_classes
conf_mat = torch.zeros((C, C), dtype=torch.int64)

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        preds = logits.argmax(dim=1)

        preds_cpu = preds.cpu()
        labels_cpu = labels.cpu()

        for t, p in zip(labels_cpu, preds_cpu):
            conf_mat[t.long(), p.long()] +=1

#Per-class accuracy

true_counts = conf_mat.sum(dim=1)
correct_counts = conf_mat.diag()
per_class_acc = correct_counts.float() / true_counts.clamp(min=1).float()

print("\nPer-class accuracy (val):")
for i, acc in enumerate(per_class_acc):
    print(f"{kept_class_names[i]}: {acc.item():.3f} ({int(correct_counts[i])}/{int(true_counts[i])})")

overall_acc = correct_counts.sum().float() / true_counts.sum().clamp(min=1).float()
print(f"\nOverall val accuracy (from confusion matrix): {overall_acc.item():.3f}")

#Show common confusions

print("\nTop Confusions (true -> predicted):")
conf_mat_no_diag = conf_mat.clone()
conf_mat_no_diag.fill_diagonal_(0)

flat = conf_mat_no_diag.flatten()
topk = min(10, flat.numel())
vals, idxs = torch.topk(flat, k=topk)

shown = 0

for v, idx in zip(vals.tolist(), idxs.tolist()):
    if v == 0:
        continue
    true_idx = idx // C
    pred_idx = idx % C
    print(f"{kept_class_names[true_idx]} -> {kept_class_names[pred_idx]}: {v}")
    shown +=1
if shown ==0:
    print("(none)")


