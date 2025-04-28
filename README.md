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
    ToTensorV2()
```
