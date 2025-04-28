import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, CoarseDropout
from albumentations.pytorch import ToTensorV2 
# from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
import datasets
from albumentations import Normalize
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

dataset_name = "matthieulel/galaxy10_decals"
galaxy_dataset = datasets.load_dataset(dataset_name)

# Define class names based on the dataset card
class_names = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"
]

# Create a dictionary for easy lookup
label2name = {i: name for i, name in enumerate(class_names)}
name2label = {name: i for i, name in enumerate(class_names)}

num_classes = len(class_names)
print(f"\nNumber of classes: {num_classes}")
print("Class names:", class_names)

class GalaxyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        image_np = np.array(image)

        if self.transform:
            transformed = self.transform(image=image_np)
            image = transformed['image']
        else:
            image = torch.transforms.ToTensor()(image)

        if isinstance(label, str):
            label = int(label)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

train_transform = Compose([
    # Rotate(limit=45, p=0.7),
    # HorizontalFlip(p=0.5),
    # RandomBrightnessContrast(p=0.3),
    # CoarseDropout(num_holes_range=(7,8), hole_height_range=(15,16), hole_width_range=(15, 16), p=0.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

    ToTensorV2()
])

test_transform = Compose([
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_dataset = GalaxyDataset(galaxy_dataset['train'], transform=train_transform)
test_dataset = GalaxyDataset(galaxy_dataset['test'], transform=test_transform)

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

sample_images, sample_labels = next(iter(train_loader))
print(f"Image batch shape: {sample_images.shape}")
print(f"Label batch shape: {sample_labels.shape}")

def epoch_train(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    accuracy.reset()
    
    for images, labels in loader:
        if isinstance(labels, list):
            labels = torch.tensor(labels)
            
        images, labels = images.to(device), labels.to(device)


        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs) 
        loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1) 
        accuracy.update(preds, labels)
    
    avg_loss = total_loss / len(loader.dataset)
    avg_acc = accuracy.compute()
    return avg_loss, avg_acc

def epoch_test(loader, model, criterion, epoch=0):
    model.eval()
    total_loss = 0.0
    accuracy.reset()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            if isinstance(labels, list):
                labels = torch.tensor(labels)
        
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            accuracy.update(preds, labels)
    
    plot_confusion_matrix(all_labels, all_preds, epoch)
    
    avg_loss = total_loss / len(loader.dataset)
    avg_acc = accuracy.compute()
    return avg_loss, avg_acc

def plot_confusion_matrix(all_labels, all_preds, epoch):
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title(f'cf  (Epoch {epoch})')
    
    filename = f'confusion_matrix_epoch_{epoch:03d}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close() 


def train(train_loader, test_loader, model, criterion, optimizer, epochs=50):
    model = model.to(device)
    
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss, train_acc = epoch_train(train_loader, model, criterion, optimizer)
        test_loss, test_acc = epoch_test(test_loader, model, criterion, epoch=epoch)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
# model = timm.create_model('resnet50', pretrained=True, num_classes=10)
# model = model.to(device)

 #https://huggingface.co/docs/transformers/model_doc/swinv2

from transformers import Swinv2ForImageClassification

model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=10, ignore_mismatched_sizes=True
)

opt = torch.optim.AdamW(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
train(train_loader, test_loader, model, criterion, opt, epochs=20)