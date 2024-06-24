from ultralytics import YOLO
import cv2
import os

folder_files = 'path to folder with images'
os.makedirs("output", exist_ok=True)

model = YOLO('path to best .bt file')

def files_list(directory_path):
    files_and_dirs = os.listdir(directory_path)
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory_path, f))]
    return files


for file in files_list(folder_files):
    img = cv2.imread(folder_files + "/" + file)
    results = model(img)
    annotated_img = results[0].plot()
    cv2.imwrite(f"output/{file}", annotated_img)


print(f'Предсказание сохранено в output')
