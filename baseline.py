import cv2
import json

from Detector.Yolo_detector import *
from numba import cuda 
from timeit import default_timer as timer  

classes = ['Cam_nguoc_chieu', 'Cam_dung_va_do', 'Cam_re', 'Gioi_han_toc_do', 'Cam_con_lai', 'Nguy_hiem', 'Hieu_lenh']
Model_path = "./Detector/cfg-model/yolov4_ver5.2.weights"
Cofig_path = "./Detector/cfg-model/yolov4_ver5.cfg"
Detector = YOLO_Detector(classes, Model_path, Cofig_path)

def WriteFile(data, image_id, class_ids, boxes, scores):
    for box in range(len(boxes)):
        print('image_id: ', image_id, 'category_id: ', class_ids[box]+1, 'bbox: ', boxes[box], 'score: ', scores[box])
        data.append({
            'image_id': int(image_id),
            'category_id': int(class_ids[box] + 1),
            'bbox': boxes[box],
            'score': scores[box]
        })

if __name__ == "__main__":
    # if os.path.exists('./output/SubmitFilev5.0.json'):
    #     os.remove('./output/SubmitFilev5.0.json')
    start = timer()
    cuda.select_device(0)
    data = []
    for image in glob.glob('./input/images/*.png'):
        ID = image.split('\\')[1].split('.')[0]
        class_ids, boxes, confidences = Detector.predict(cv2.imread(image), ID)
        WriteFile(data, ID, class_ids, boxes, confidences)
    with open('./output/SubmitFilev5.2.2.json', 'a') as Sf:
        json.dump(data, Sf)

    cuda.close()
    print("time:", timer()-start)