from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2, os
from yolov8 import YOLOv8
from ultralytics import YOLO
import numpy as np
import time
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

class_name=["Flute", "Guitar", "Drum", "Piano", "Saxophone", "Trumpet", "Watercraft", "Rose", "Houseplant",
            "Flowerpot", "Baseball bat", "Ball", "Tennis racket", "Tennis ball", "Table tennis racket", "Cricket ball",
            "Bicycle helmet", "Helmet", "Football helmet", "Book", "Shirt", "Jacket", "Footwear", "Watch", "Handbag", 
            "Luggage and bags", "Sandal", "Necklace", "Table", "Chair", "Couch", "Wall clock", "Lamp", "Sofa bed", "Bed",
            "Stool", "Cat", "Dog", "Mouse", "Car", "Motorcycle",  
            "Television", "Mobile phone", "Laptop", "Hair dryer", "Washing machine", "Microwave oven",
            "Gas stove", "Toaster", "Kettle", "Coffee table", "Coffeemaker", "Grinder", "Clock", "Mechanical fan", 
            "Ceiling fan", "Dishwasher", "Printer", "Cassette deck", "Light bulb", "Computer keyboard", 
            "Computer mouse", "Headphones", "Corded phone",
            "Banjo"
            ]

onnx_model_path = "models/best.onnx"
yolov8_detector = YOLOv8(onnx_model_path, conf_thres=0.2, iou_thres=0.3)

pytorch_model_path = 'models/best.pt'
pytorch_model = YOLO(pytorch_model_path)

result_folder_name = "result_folder"
os.makedirs(result_folder_name, exist_ok=True)


def get_relative_coordinate(box_list, img_height, img_width):
    return [
        round(float(box_list[0]) / img_width, 3),
        round(float(box_list[1]) / img_height, 3),
        round(float(box_list[2]) / img_width, 3),
        round(float(box_list[3]) / img_height, 3)
    ]


def write_detect_image(filename, bbox_img):
    cv2.imwrite(os.path.join(result_folder_name, filename), bbox_img)
            

def detect_objects_onnx(img):
    boxes, scores, class_ids = yolov8_detector(img)
    draw_img = yolov8_detector.draw_detections(img)
    return (boxes, scores, class_ids), draw_img


def onnx_model_detection(img, img_height, img_width):
    
    multi_objects = []

    (boxes, conf, class_id), draw_bbox_img = detect_objects_onnx(img)
            
    for box, cnf, cls in zip(boxes, conf, class_id):
        x1, y1, x2, y2 = get_relative_coordinate(box, img_height, img_width)
        
        object = {
            "bbox": {"x1":max(x1,0), "y1":max(y1,0), "x2":x2, "y2":y2},
            "confidence": round(float(cnf), 2),
            "class_name": class_name[int(cls)],
        }
        multi_objects.append(object)

    return multi_objects, draw_bbox_img


def pytorch_model_detection(img, img_height, img_width):
    
    multi_objects = []

    img_copy = img.copy() 
    results = pytorch_model.predict(img_copy, device='cpu')
    
    for result in results:
        boxes = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        boxes = boxes.tolist()
        for box, conf_val, cls_val in zip(boxes, conf, cls):
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            x1, y1, x2, y2 = get_relative_coordinate(box, img_height, img_width)
        
            object = {
                "bbox": {"x1":max(x1,0), "y1":max(y1,0), "x2":x2, "y2":y2},
                "confidence": round(conf_val.item(), 2),
                "class_name": class_name[int(cls_val)],
            }
            multi_objects.append(object)

            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name[int(cls_val)]} ({round(conf_val.item(), 2)})"
            cv2.putText(img_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return multi_objects, img_copy


@app.route('/detect/', methods=['POST'])
def start_process():
    try:
        response = {}
        
        start_time = time.time()

        write_img = request.args.get("write_image")

        acc_type = request.args.get("accurate_type")

        file = request.files['image']
        response["filename"] = file.filename

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        img_height, img_width, _ = img.shape
        response["img_height"] = img_height
        response["img_width"] = img_width

        if acc_type == "False":
            multi_objects, draw_bbox_img = onnx_model_detection(img, img_height, img_width)
        else:
            multi_objects, draw_bbox_img = pytorch_model_detection(img, img_height, img_width)

        response["objects"] = multi_objects
        
        print(f"Time Taken: {round((time.time()-start_time), 2)} seconds")

        if write_img == "True":
            write_detect_image(file.filename, draw_bbox_img)

        return response
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port='8080', threaded=True, debug=False)
