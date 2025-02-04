import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
class_names = ["breach", "kayo"]

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
print(os.path.exists("model.pth"))
model.load_state_dict(torch.load(r"C:\Users\mzhum\OneDrive\Pulpit\ai\project_track\model.pth"))

model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_motion(frame1, frame2, threshold=30, min_area=500):

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) 
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def classify_object(frame, bbox):
    x, y, w, h = bbox
    object_img = frame[y-w:y+h, x-w:x+w]
    object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(object_img).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)
    
    class_label = class_names[predicted_class.item()]
    class_prob = probabilities[0, predicted_class].item()
    return class_label, class_prob

# Обработка видео
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {input_path}")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None

    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Ошибка: не удалось считать кадр (конец видео или повреждённый файл).")
            break

        if prev_frame is None:
            prev_frame = frame
            continue

        contours = detect_motion(prev_frame, frame)
        prev_frame = frame.copy()

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            class_label, class_prob = classify_object(frame, (x, y, w, h))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_label} ({class_prob:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if out is None:
            try:
                h, w, _ = frame.shape
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
                if not out.isOpened():
                    print("Ошибка: VideoWriter не удалось открыть!")
                    out = None
            except Exception as e:
                print(f"Ошибка при инициализации VideoWriter: {e}")
                out = None

        if out is not None:
            out.write(frame)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    
    if out is not None:
        out.release()
    
    cv2.destroyAllWindows()



# Запуск обработки
input_video = r"C:\Users\mzhum\OneDrive\Pulpit\ai\project_track\video_val/track_video.mp4"
output_video = "output.avi"
process_video(input_video, output_video)
