This reposiotry is about what i have done in my Internship at KJSSE, Mumbai.
It focuses on remote sensing land cover classification using deep learning models such as UNet, SegNet, and DeepLabV3.
The goal is to automate satellite image segmentation and analyze seasonal changes in regions like water, vegetation, urban, forest, and roads.


A. Models Used:
1. UNet – Encoder–decoder architecture with skip connections for precise pixel segmentation.
2. SegNet – VGG-based encoder–decoder network using pooling indices for smoother feature recovery.
3. DeepLabV3 – ResNet backbone with atrous spatial pyramid pooling (ASPP) for detailed boundary detection.

B. Dataset Link: https://www.kaggle.com/datasets/khushiipatni/satellite-image-and-mask/data

C. How to Run:
1. Download dataset
2. Make changes in train.py for dataset path
3. Make changes in test.py for inferencing
4. On terminal
   - pip install -r requirements.txt        # Install dependencies  
   - python3 train.py                       # Train all 3 models (UNet, SegNet, DeepLabV3)  
   - python3 test.py                        # Compare April vs May and generate results  
     
