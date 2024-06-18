from ultralytics import YOLO
import cv2

model = YOLO("path to runs folder/weights/best.pt")

img = cv2.imread("path to detect image")

results = model(img)

# Получить аннотированное изображение
annotated_img = results[0].plot()

# Сохранить изображение с предсказаниями
output_path = 'output path .jpg'
cv2.imwrite(output_path, annotated_img)

print(f'Предсказание сохранено в {output_path}')
