# load libraries
from ultralytics import YOLO
import supervision as sv
from supervision import Detections
from PIL import Image
import numpy as np

# download model
# model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO('/app/model/model.pt')

# # inference
# image_path = "/app/videos/Univ_Building/00000995.jpg"
# results = model(Image.open(image_path))[0]
# detections = Detections.from_ultralytics(results)

# bounding_box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# labels = [
#     model.model.names[class_id]
#     for class_id
#     in detections.class_id
# ]

# annotated_image = bounding_box_annotator.annotate(
#     scene=Image.open(image_path), detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)

# sv.plot_image(annotated_image)
# boxes = detections.xyxy[0]
# print(boxes)
# print(results.boxes)


tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    print(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    print(labels)

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

sv.process_video(
    source_path="/app/videos/DTC_Video_To_Test2.mp4",
    target_path="/app/videos/DTC_Video_To_Test2_tracking.mp4",
    callback=callback
)