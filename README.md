-Models Used:
UNet – Encoder–decoder architecture with skip connections for precise pixel segmentation.
SegNet – VGG-based encoder–decoder network using pooling indices for smoother feature recovery.
DeepLabV3 – ResNet backbone with atrous spatial pyramid pooling (ASPP) for detailed boundary detection.

-Dataset Link: https://www.kaggle.com/datasets/khushiipatni/satellite-image-and-mask/data

-How to Run:
1. Download dataset
2. Make changes in train.py for dataset path
3. Make changes in test.py for inferencing
4. On terminal
   - pip install -r requirements.txt        # Install dependencies  
   - python3 train.py                       # Train all 3 models (UNet, SegNet, DeepLabV3)  
   - python3 test.py                        # Compare April vs May and generate results  
     
