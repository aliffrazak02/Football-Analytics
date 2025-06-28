from ultralytics import YOLO

model = YOLO("yolov8s") 
results = model.predict(source="input_videos/08fd33_4.mp4", save=True, project="output_video", name="my_run") 
print(results[0])  
print("----------------------------------------")
for box in results[0].boxes:
    print(box)