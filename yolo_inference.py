from ultralytics import YOLO # Import the YOLO class from the Ultralytics library
# Load the pretrained YOLOv8 small model (yolov8s)
model = YOLO("yolov8s") 

# Run inference on the input video
# - save=True: save the output video with detection overlays
# - project/name: set custom output directory
results = model.predict(source="input_videos/08fd33_4.mp4", save=True, project="output_video", name="my_run") 

# Print summary of the first result (e.g., frame-level detections)
print(results[0])  
print("----------------------------------------")

# Iterate over the detected bounding boxes in the first frame and print them
for box in results[0].boxes:
    print(box)