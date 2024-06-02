# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import supervision as sv
from supervision import Detections
from PIL import Image

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
image_path = "/app/videos/Univ_Building/00000995.jpg"
results = model(Image.open(image_path))[0]
detections = Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    model.model.names[class_id]
    for class_id
    in detections.class_id
]

annotated_image = bounding_box_annotator.annotate(
    scene=Image.open(image_path), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(annotated_image)
boxes = detections.xyxy[0]
print(boxes)
print(results.boxes)