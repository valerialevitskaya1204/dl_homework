import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg19, VGG19_Weights
from albumentations import Normalize
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, CoarseDropout
from albumentations.pytorch import ToTensorV2 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datasets
from tqdm import tqdm

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def changedClassifierLayer(model, modelName, N_CLASSES=10):
    for param in model.parameters():
      param.requires_grad = False

    if modelName == "DenseNet121":
      num_input = model.classifier.in_features

    elif modelName == "ResNet50":
      num_input = model.fc.in_features

    elif modelName == "EfficientNet-V2-M" or modelName == "AlexNet":
      num_input = model.classifier[1].in_features

    elif modelName == "VGG19" or modelName == "VGG16":
      num_input = model.classifier[0].in_features

    classifier = nn.Sequential(
      nn.Linear(num_input, 256),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(128, N_CLASSES),
      nn.LogSoftmax(dim=1)
    )

    if modelName == "ResNet50":
      model.fc = classifier
    else:
      model.classifier = classifier

efficientnet_weights_path = 'models/EfficientNet-V2-M.pth'
densenet_weights_path = 'models/DenseNet121.pth'
resnet_weights_path = 'models/ResNet50.pth'
alexnet_weights_path = 'models/AlexNet.pth'
vgg16_weights_path = 'models/VGG16.pth'
vgg19_weights_path = 'models/VGG19.pth'

efficientnetV2M_model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
densenet_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
alexnet_model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

class EnsembleModel(nn.Module):
    def __init__(self, model_list, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.weights = weights

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)  
            outputs.append(output)
        
        if self.weights is None:
            ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        else:
            weighted_outputs = torch.stack([w * output for w, output in zip(self.weights, outputs)])
            ensemble_output = torch.sum(weighted_outputs, dim=0)

        return ensemble_output

models_list = [
    # efficientnetV2M_model.to(device),
    # densenet_model.to(device),
    resnet_model.to(device),
    alexnet_model.to(device),
    vgg16_model.to(device),
    vgg19_model.to(device)
]

ensemble_model = EnsembleModel(models_list)

model = ensemble_model.to(device)
parameters_to_optimize = []

for m in models_list:
    parameters_to_optimize += list(filter(lambda p: p.requires_grad, m.parameters()))

opt = torch.optim.AdamW(parameters_to_optimize, lr=0.005)
criterion = nn.CrossEntropyLoss()

import seaborn as sns

accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

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

train(train_loader, test_loader, model, criterion, opt, epochs=20)

