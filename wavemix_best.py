import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch, time
from torchinfo import summary
from wavemix.classification import WaveMix
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torchmetrics.classification import Accuracy
from PIL import Image
import h5py
import numpy as np
import torch

torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True  

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

logger.info("Loading dataset from Galaxy10_DECals.h5")
try:
    with h5py.File('Galaxy10_DECals.h5', 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])
    logger.info(f"Loaded dataset with {len(images)} samples")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

logger.info("Converting labels to one-hot encoding")
labels = torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.long), num_classes=10)
labels = labels.float()  
images = images.astype(np.float32)

logger.info("Splitting dataset into train/test sets")
try:
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels = images[train_idx], labels[train_idx]
    test_images, test_labels = images[test_idx], labels[test_idx]
    logger.info(f"Train set: {len(train_images)} samples, Test set: {len(test_images)} samples")
except Exception as e:
    logger.error(f"Dataset split failed: {e}")
    raise

class CustomImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        logger.debug("Initializing CustomImageDataset")
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = self.data[idx]
            label = self.labels[idx]
            if self.transform:
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transform(img)
            return img, label
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise

logger.info("Setting up data transformations")
transform_train = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
])

try:
    trainset = CustomImageDataset(train_images, train_labels, transform=transform_train)
    testset = CustomImageDataset(test_images, test_labels, transform=transforms.ToTensor())
    logger.info(f"Train dataset size: {len(trainset)}, Test dataset size: {len(testset)}")
except Exception as e:
    logger.error(f"Dataset creation failed: {e}")
    raise

logger.info("Initializing WaveMix model")
try:
    model = WaveMix(
        num_classes=10,
        depth=16,
        mult=2,
        ff_channel=192,
        final_dim=192,
        dropout=0.5,
        level=3,
        initial_conv='pachify',
        patch_size=4
    )
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    raise

logger.info("Loading pretrained weights")
try:
    url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/galaxy10/galaxy_95.42.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    logger.info("Pretrained weights loaded successfully")
except Exception as e:
    logger.error(f"Failed to load pretrained weights: {e}")
    raise

model.to(device)
logger.info("Model moved to device")

try:
    logger.info("Model summary:")
    model_summary = summary(model, (1,3,256,256))
    logger.info(str(model_summary))
except Exception as e:
    logger.warning(f"Could not generate model summary: {e}")

batch_size = 32
logger.info(f"Creating data loaders with batch size {batch_size}")
try:
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True, 
        prefetch_factor=2, persistent_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True, 
        prefetch_factor=2, persistent_workers=2
    )
    logger.info("Data loaders created successfully")
except Exception as e:
    logger.error(f"Data loader creation failed: {e}")
    raise

top1_acc = Accuracy(task="multiclass", num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")
logger.info("Training components initialized")

top1 = []
top5 = []
traintime = []
testtime = []
counter = 0

logger.info("Starting training with AdamW optimizer")
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
PATH = 'galaxy.pth'
epoch = 0

while counter < 10:
    logger.info(f"Starting epoch {epoch+1}")
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    model.train()

    try:
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for data in tepoch:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                with torch.cuda.amp.autocast():
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
                epoch_accuracy += acc / len(trainloader)
                epoch_loss += loss / len(trainloader)
                tepoch.set_postfix_str(f"loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}")
    except Exception as e:
        logger.error(f"Training failed at epoch {epoch+1}: {e}")
        raise

    correct_1 = 0
    c = 0
    model.eval()
    t1 = time.time()
    
    try:
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                correct_1 += top1_acc(outputs, labels.argmax(dim=1))
                c += 1
    except Exception as e:
        logger.error(f"Evaluation failed at epoch {epoch+1}: {e}")
        raise

    logger.info(f"Epoch {epoch+1} results - Top 1: {correct_1*100/c:.2f}% - Train Time: {t1-t0:.2f}s - Test Time: {time.time()-t1:.2f}s")
    
    top1.append(correct_1*100/c)
    traintime.append(t1-t0)
    testtime.append(time.time()-t1)
    counter += 1
    epoch += 1
    
    if float(correct_1*100/c) >= float(max(top1)):
        try:
            torch.save(model.state_dict(), PATH)
            logger.info(f"New best model saved at epoch {epoch} with accuracy {correct_1*100/c:.2f}%")
            counter = 0
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

logger.info("Switching to SGD optimizer")
try:
    model.load_state_dict(torch.load(PATH))
    logger.info("Loaded best model weights for SGD training")
except Exception as e:
    logger.error(f"Failed to load model weights: {e}")
    raise

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
counter = 0
epoch = 0

while counter < 20:
    logger.info(f"Starting SGD epoch {epoch+1}")
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    model.train()

    try:
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for data in tepoch:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                with torch.amp.autocast("cuda"):
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
                epoch_accuracy += acc / len(trainloader)
                epoch_loss += loss / len(trainloader)
                tepoch.set_postfix_str(f"loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}")
    except Exception as e:
        logger.error(f"SGD training failed at epoch {epoch+1}: {e}")
        raise

    correct_1 = 0
    c = 0
    model.eval()
    t1 = time.time()
    
    try:
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                correct_1 += top1_acc(outputs, labels.argmax(dim=1))
                c += 1
    except Exception as e:
        logger.error(f"SGD evaluation failed at epoch {epoch+1}: {e}")
        raise

    logger.info(f"Epoch {epoch+1} results - Top 1: {correct_1*100/c:.2f}% - Train Time: {t1-t0:.2f}s - Test Time: {time.time()-t1:.2f}s")
    
    top1.append(correct_1*100/c)
    traintime.append(t1-t0)
    testtime.append(time.time()-t1)
    counter += 1
    epoch += 1
    
    if float(correct_1*100/c) >= float(max(top1)):
        try:
            torch.save(model.state_dict(), PATH)
            logger.info(f"New best model saved at epoch {epoch} with accuracy {correct_1*100/c:.2f}%")
            counter = 0
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

logger.info("Training completed")
logger.info(f"Best Top 1 Accuracy: {max(top1):.2f}%")
logger.info(f"Minimum Train Time: {min(traintime):.2f}s")
logger.info(f"Minimum Test Time: {min(testtime):.2f}s")