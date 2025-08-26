import cv2
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO("yolov5s.pt")  # Auto-downloads if not present

# Colors for COCO dataset classes (80 classes)
colors = {
    'person': (0, 255, 0),        # Green
    'car': (255, 0, 0),           # Blue
    'truck': (0, 0, 255),         # Red
    'motorcycle': (255, 255, 0),  # Yellow
    'bus': (255, 0, 255),         # Magenta
    'bicycle': (0, 255, 255),     # Cyan
    'train': (128, 0, 128),       # Purple
    'traffic light': (255, 165, 0), # Orange
    'fire hydrant': (255, 69, 0), # Orange Red
    'stop sign': (220, 20, 60),   # Crimson
    'parking meter': (128, 128, 0), # Olive
    'bench': (139, 69, 19),       # Brown
    'bird': (0, 128, 255),        # Light Blue
    'cat': (160, 32, 240),        # Purple Orchid
    'dog': (255, 20, 147),        # Deep Pink
    'horse': (184, 134, 11),      # Dark Goldenrod
    'sheep': (0, 255, 127),       # Spring Green
    'cow': (255, 140, 0),         # Dark Orange
    'elephant': (128, 0, 0),      # Maroon
    'bear': (75, 0, 130),         # Indigo
    'zebra': (47, 79, 79),        # Dark Slate Gray
    'giraffe': (218, 112, 214),   # Orchid
    'backpack': (0, 191, 255),    # Deep Sky Blue
    'umbrella': (199, 21, 133),   # Medium Violet Red
    'handbag': (255, 228, 181),   # Moccasin
    'tie': (0, 128, 0),           # Dark Green
    'suitcase': (70, 130, 180),   # Steel Blue
    'frisbee': (255, 215, 0),     # Gold
    'skis': (46, 139, 87),        # Sea Green
    'snowboard': (123, 104, 238), # Medium Slate Blue
    'sports ball': (255, 99, 71), # Tomato
    'kite': (0, 206, 209),        # Dark Turquoise
    'baseball bat': (205, 133, 63), # Peru
    'baseball glove': (139, 0, 0), # Dark Red
    'skateboard': (255, 182, 193), # Light Pink
    'surfboard': (95, 158, 160),  # Cadet Blue
    'tennis racket': (72, 61, 139), # Dark Slate Blue
    'bottle': (34, 139, 34),      # Forest Green
    'wine glass': (255, 250, 205), # Lemon Chiffon
    'cup': (210, 105, 30),        # Chocolate
    'fork': (128, 128, 128),      # Gray
    'knife': (112, 128, 144),     # Slate Gray
    'spoon': (0, 0, 128),         # Navy
    'bowl': (173, 216, 230),      # Light Blue
    'banana': (255, 255, 102),    # Light Yellow
    'apple': (144, 238, 144),     # Light Green
    'sandwich': (255, 222, 173),  # Navajo White
    'orange': (255, 165, 0),      # Orange
    'broccoli': (85, 107, 47),    # Dark Olive Green
    'carrot': (255, 127, 80),     # Coral
    'hot dog': (139, 0, 139),     # Dark Magenta
    'pizza': (233, 150, 122),     # Dark Salmon
    'donut': (255, 105, 180),     # Hot Pink
    'cake': (106, 90, 205),       # Slate Blue
    'chair': (160, 82, 45),       # Sienna
    'couch': (0, 139, 139),       # Dark Cyan
    'potted plant': (107, 142, 35), # Olive Drab
    'bed': (119, 136, 153),       # Light Slate Gray
    'dining table': (255, 228, 196), # Bisque
    'toilet': (255, 250, 250),    # Snow White
    'tv': (70, 130, 180),         # Steel Blue
    'laptop': (138, 43, 226),     # Blue Violet
    'mouse': (0, 250, 154),       # Medium Spring Green
    'remote': (244, 164, 96),     # Sandy Brown
    'keyboard': (255, 69, 0),     # Orange Red
    'cell phone': (0, 100, 0),    # Dark Green
    'microwave': (205, 92, 92),   # Indian Red
    'oven': (139, 69, 19),        # Saddle Brown
    'toaster': (255, 239, 213),   # Papaya Whip
    'sink': (135, 206, 250),      # Light Sky Blue
    'refrigerator': (65, 105, 225), # Royal Blue
    'book': (205, 133, 63),       # Peru
    'clock': (255, 20, 147),      # Deep Pink
    'vase': (186, 85, 211),       # Medium Orchid
    'scissors': (46, 139, 87),    # Sea Green
    'teddy bear': (222, 184, 135), # BurlyWood
    'hair drier': (176, 224, 230), # Powder Blue
    'toothbrush': (218, 165, 32),  # GoldenRod
}

def process_and_save_video():
    video_path = input("Enter the path to your video file: ").strip('"')
    output_path = 'output_with_detection.mp4'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Video Writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 model on the frame
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
