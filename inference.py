import os
import cv2
from pathlib import Path
from ultralytics import YOLO

def get_detector(args):
    weights_path = args.weights
    classes_path = args.classes
    source_path = args.source
    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
    assert os.path.isfile(source_path), f"There's no source file with name {weights_path}"

    if args.image:
        image = cv2.imread(source_path)
        h, w = image.shape[:2]
    elif args.video:
        cap = cv2.VideoCapture(source_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))'
        detector = YOLO(weights_path) # weights in .pt
    return detector

def predict(chosen_model, img, classes=[], conf = 0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def inference_on_image(args):
    print("[INFO] Intialize Model")
    detector = get_detector(args)
    image = cv2.imread(args.source)

    print("[INFO] Inference Image")
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)

    output_path = f"output/{Path(args.source).name}"
    print(f"[INFO] Saving result on {output_path}")
    cv2.imwrite(output_path, image)

    if args.show:
        cv2.imshow("Result", image)
        cv2.waitKey(0)

def save_results(results):
    print(results[0].boxes)
    pass

def inference_on_video(args):
    print("[INFO] Intialize Model")
    #detector = get_detector(args)

    cap = cv2.VideoCapture(args.source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter('output/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (w, h))
    
    model = YOLO(args.weights)

    print("[INFO] Inference on Video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #detections = detector.detect(frame)
        #detector.draw_detections(frame, detections=detections)
        frame, results = predict_and_detect(model, frame, classes=None, conf=0.3, rectangle_thickness=2, text_thickness=1)
        save_results(results)
        writer.write(frame)
        cv2.imshow("Result", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    print("[INFO] Finish. Saving result to output/result.avi")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Argument for YOLOv9 Inference using ONNXRuntime")

    parser.add_argument("--source", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--weights", type=str, required=True, help="Path to yolov9 onnx file")
    parser.add_argument("--classes", type=str, required=False, help="Path to list of class in yaml file")
    parser.add_argument("--score-threshold", type=float, required=False, default=0.1)
    parser.add_argument("--conf-threshold", type=float, required=False, default=0.4)
    parser.add_argument("--iou-threshold", type=float, required=False, default=0.4)
    parser.add_argument("--image", action="store_true", required=False, help="Image inference mode")
    parser.add_argument("--video", action="store_true", required=False)
    parser.add_argument("--show", required=False, type=bool, default=True, help="Show result on pop-up window")

    args = parser.parse_args()

    if args.image:
        inference_on_image(args=args)
    elif args.video:
        inference_on_video(args=args)
    else:
        raise ValueError(
            "You can't process the result because you have not define the source type (video or image) in the argument")
