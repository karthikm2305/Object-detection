import cv2
from ultralytics import YOLO


model = YOLO("yolov5s.pt")  

colors = {
    'person': (0, 255, 0),        
    'car': (255, 0, 0),        
    'truck': (0, 0, 255),         
    'motorcycle': (255, 255, 0), 
    'bus': (255, 0, 255),         
    'bicycle': (0, 255, 255),     
    'train': (128, 0, 128),       
    'traffic light': (255, 165, 0), 
    'fire hydrant': (255, 69, 0),
    'stop sign': (220, 20, 60),   
    'parking meter': (128, 128, 0), 
    'bench': (139, 69, 19),      
    'bird': (0, 128, 255),       
    'cat': (160, 32, 240),        
    'dog': (255, 20, 147),        
    'horse': (184, 134, 11),   
    'sheep': (0, 255, 127),      
    'cow': (255, 140, 0),         
    'elephant': (128, 0, 0),     
    'bear': (75, 0, 130),       
    'zebra': (47, 79, 79),       
    'giraffe': (218, 112, 214),   
    'backpack': (0, 191, 255),   
    'umbrella': (199, 21, 133),  
    'handbag': (255, 228, 181),   
    'tie': (0, 128, 0),           
    'suitcase': (70, 130, 180),  
    'frisbee': (255, 215, 0),    
    'skis': (46, 139, 87),       
    'snowboard': (123, 104, 238), 
    'sports ball': (255, 99, 71), 
    'kite': (0, 206, 209),       
    'baseball bat': (205, 133, 63), 
    'baseball glove': (139, 0, 0), 
    'skateboard': (255, 182, 193),
    'surfboard': (95, 158, 160),  
    'tennis racket': (72, 61, 139),
    'bottle': (34, 139, 34),     
    'wine glass': (255, 250, 205),
    'cup': (210, 105, 30),        
    'fork': (128, 128, 128),    
    'knife': (112, 128, 144),     
    'spoon': (0, 0, 128),         
    'bowl': (173, 216, 230),      
    'banana': (255, 255, 102),   
    'apple': (144, 238, 144),     
    'sandwich': (255, 222, 173),  
    'orange': (255, 165, 0),      
    'broccoli': (85, 107, 47),    
    'carrot': (255, 127, 80),     
    'hot dog': (139, 0, 139),    
    'pizza': (233, 150, 122),    
    'donut': (255, 105, 180),     
    'cake': (106, 90, 205),      
    'chair': (160, 82, 45),       
    'couch': (0, 139, 139),      
    'potted plant': (107, 142, 35), 
    'bed': (119, 136, 153),      
    'dining table': (255, 228, 196),
    'toilet': (255, 250, 250),  
    'tv': (70, 130, 180),         
    'laptop': (138, 43, 226),     
    'mouse': (0, 250, 154),       
    'remote': (244, 164, 96),     
    'keyboard': (255, 69, 0),     
    'cell phone': (0, 100, 0),    
    'microwave': (205, 92, 92),   
    'oven': (139, 69, 19),        
    'toaster': (255, 239, 213),   
    'sink': (135, 206, 250),      
    'refrigerator': (65, 105, 225), 
    'book': (205, 133, 63),      
    'clock': (255, 20, 147),    
    'vase': (186, 85, 211),      
    'scissors': (46, 139, 87),   
    'teddy bear': (222, 184, 135),
    'hair drier': (176, 224, 230), 
    'toothbrush': (218, 165, 32), 
}

def process_and_save_video():
    video_path = input("Enter the path to your video file: ").strip('"')
    output_path = 'output_with_detection.mp4'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

      
        results = model(frame)

        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            label = model.names[cls]

            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Processed video saved as {output_path}")


if __name__ == "__main__":
    process_and_save_video()

