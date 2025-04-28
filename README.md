# dl_homework

Repository of our results for dl homework

Levitskaia Valeriia 
Evgenia Shustova


1. SWIN models

   Tried to train swin model https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/swinv2#transformers.Swinv2ForImageClassification
   with following data augmentation
```python
   train_transform = Compose([
    Rotate(limit=45, p=0.7),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.3),
    CoarseDropout(num_holes_range=(7,8), hole_height_range=(15,16), hole_width_range=(15, 16), p=0.5),
    Normalize(mean=processor.image_mean, std=processor.image_std),
    ToTensorV2()])
```

The best result is 0.86. Checkpoint in repository (swin_3)

2. Ensemble models

   ```python
   efficientnet_weights_path = 'models/EfficientNet-V2-M.pth'
densenet_weights_path = 'models/DenseNet121.pth'
resnet_weights_path = 'models/ResNet50.pth'
alexnet_weights_path = 'models/AlexNet.pth'
vgg16_weights_path = 'models/VGG16.pth'
vgg19_weights_path = 'models/VGG19.pth'
```
with mean voting


3. WaveMix

WaveMix model gave desired accuracy (91.02) up to 1 epoch. Code to train and log results is in the repositiry with chekpoint.

4. Diffferent data augmentations
   Sequentially:
    - ElasticTransform from https://ieeexplore.ieee.org/document/1227801
    - ChannelWiseDropout from AstroAugmentations
    - ShiftScaleRotate for more control
    - Flip along x or y axis


