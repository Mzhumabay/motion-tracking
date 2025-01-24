import torch
import cv2
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

save_dir_breach = "breach"
save_dir_kayo = "kayo"
save_dir_noise = "noise"

os.makedirs(save_dir_breach, exist_ok=True)
os.makedirs(save_dir_kayo, exist_ok=True)
os.makedirs(save_dir_noise, exist_ok=True)

videos_breach = ["breach_back", "breach_left", "breach_right", "breach_straight"]
videos_kayo = ["kayo_back", "kayo_left", "kayo_right", "kayo_straight"]
test = ["breach_straight"]

def make_dataset(caps_list, save_dir):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

    for cap1 in caps_list:
        cap = cv2.VideoCapture(f"project_track/video_val/{cap1}.mp4")
        frame_count = 0
        person_count = 0 
        noise_count = 0   

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for bbox in results.xywh[0]:
                class_object = int(bbox[5])
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                            
                object_img = frame[y-h:y+h, x-w:x+w]

                if class_object == 0:
                    person_count += 1
                    object_filename = f"{save_dir}/{cap1}_object_{person_count}.png"
                    
                cv2.imwrite(object_filename, object_img)

            if person_count > 1600:
                break
        cap.release()


def main():

    #make_dataset(test, save_dir_breach)
    make_dataset(videos_breach, save_dir_breach)
    make_dataset(videos_kayo, save_dir_kayo)

if __name__ == "__main__":
    main()

