from ultralytics import YOLO
import cv2

model = YOLO('/home/anhalu/anhalu-data/ocr_general_core/reader/models/layout_ocr_yolov8n_v5.pt')


img = cv2.imread('/home/anhalu/anhalu-data/ocr_general_core/data/image/requests/979367fa-0107-4ea1-98cd-c27db1c8bed8_0.jpg')

pred = model(img, imgsz=640)
cv2.imshow('test', pred[0].plot())
cv2.waitKey(0)