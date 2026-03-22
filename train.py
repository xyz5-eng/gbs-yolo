import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s-GAM-bifpn.yaml')
    model.load('weights/yolov8s.pt') # loading pretrain weights
    model.train(data='dataset/mydata.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolov8s-GAM-bifpn-SIoU',
                )