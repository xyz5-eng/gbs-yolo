# Dataset configuration
    1. Put the downloaded dataset under the "dataset" folder path
    2. Open mydata.ymal, change the path to the absolute path of the working folder, and change the train, Val, and test to the relative path of the dataset image folder

# Environment configuration
    1. Copy the installation link on the pytorch official website and install the pytorch environment at: https://pytorch.org/
    2. pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    3. pip install requests==2.32.5 psutil==7.2.2 pandas==2.3.3

# Training Command
    python train.py

# Validate command
    python val.py

# Detection command
    python detect.py