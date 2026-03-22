import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yolov8s-GAM-bifpn-SIoU/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/test/images',
                  imgsz=640,
                  project='runs/detect',
                  name='yolov8s-GAM-bifpn-SIoU-rain',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )