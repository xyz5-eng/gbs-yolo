import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yolov8s-GAM-bifpn-SIoU/weights/best.pt')
    model.val(data='dataset/mydata.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8s-GAM-bifpn-SIoU',
              )