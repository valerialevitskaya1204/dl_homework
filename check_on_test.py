import logging
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datasets
from wavemix.classification import WaveMix


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('galaxy_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = "matthieulel/galaxy10_decals"
PRETRAINED_PATH = 'galaxy.pth' 
PRETRAINED_URL = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/galaxy10/galaxy_95.42.pth'

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"
]

class GalaxyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        logger.info(f"Initialized dataset with {len(self)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = np.array(item['image'])
            label = int(item['label'])

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise

train_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def load_model():
    """Load and configure the WaveMix model"""
    try:
        logger.info("Initializing WaveMix model")
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

        try:
            logger.info(f"Attempting to load local weights: {PRETRAINED_PATH}")
            state_dict = torch.load(PRETRAINED_PATH, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info("Local weights loaded successfully")
        except FileNotFoundError:
            logger.warning("Local weights not found, loading from URL")
            state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_URL, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info("Weights loaded successfully from URL")

        model.to(DEVICE)
        model.eval()
        logger.info(f"Model moved to {DEVICE} and set to evaluation mode")
        return model

    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

def generate_predictions(model, test_loader):
    """Generate predictions with logging"""
    logger.info("Starting predictions...")
    preds = []
    true_labels = []
    total_batches = len(test_loader)
    
    try:
        with torch.no_grad():
            start_time = time.time()
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                preds.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Processed {batch_idx+1}/{total_batches} batches | "
                        f"Time: {elapsed:.2f}s | "
                        f"Samples: {(batch_idx+1)*BATCH_SIZE}"
                    )

        logger.info(f"Prediction completed. Total samples: {len(preds)}")
        return preds, true_labels

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def evaluate_predictions(preds, true_labels):
    """Evaluate model performance"""
    logger.info("Starting evaluation")
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        accuracy = accuracy_score(true_labels, preds)
        logger.info(f"Test Accuracy: {accuracy:.4f}")

        logger.info("Classification Report:")
        logger.info("\n" + classification_report(true_labels, preds, target_names=CLASS_NAMES))

        cm = confusion_matrix(true_labels, preds)
        logger.info("Confusion Matrix:")
        for i, row in enumerate(cm):
            logger.info(f"{CLASS_NAMES[i]:<25}: {row}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_report(true_labels, preds, output_dict=True)
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def main():
    """Main execution flow"""
    try:
        logger.info("Loading dataset")
        galaxy_dataset = datasets.load_dataset(DATASET_NAME)

        test_dataset = GalaxyDataset(galaxy_dataset['test'], transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        logger.info(f"Test loader created with {len(test_loader)} batches")

        model = load_model()

        preds, true_labels = generate_predictions(model, test_loader)

        metrics = evaluate_predictions(preds, true_labels)
        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()