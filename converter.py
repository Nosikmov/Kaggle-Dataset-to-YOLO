import csv
import numpy as np
import cv2
import os 
import shutil
import random

main_folder     = "path to project folder"
image_folder    = 'image folder'
csv_file        = '.csv file name'
output_folder   = f"{main_folder}\\datasets"

def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height, class_id):

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    width = x2 - x1
    height = y2 - y1

    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    class_id = int(class_id) - 1
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"

def draw_square_around_mask(image, mask, classid):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotation_string = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        annotation_string += convert_to_yolo_format(x, y, x + w, y + h, image.shape[1], image.shape[0], classid)

    return annotation_string
    

def rle_to_mask(rle_str, image_shape):
    rle = list(map(int, rle_str.split()))
    predlastindex = len(rle) - 2
    if len(rle) > 2 and rle[len(rle) - 1] + rle[predlastindex] > 409600:
        rle[predlastindex] -= 1
    pixel, pixel_count = [], []
    [pixel.append(rle[i]) if i % 2 == 0 else pixel_count.append(rle[i]) for i in range(0, len(rle))]
    rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(0, len(pixel))]
    rle_mask_pixels = sum(rle_pixels, [])
    mask_img = np.zeros((image_shape[0] * image_shape[1],), dtype=int)
    mask_img[rle_mask_pixels] = 255
    mask = np.reshape(mask_img, image_shape[::-1]).T
    return mask


os.makedirs(output_folder, exist_ok=True)
os.makedirs(f"{output_folder}\\train", exist_ok=True)
os.makedirs(f"{output_folder}\\valid", exist_ok=True)
os.makedirs(f"{output_folder}\\train\\images", exist_ok=True)
os.makedirs(f"{output_folder}\\train\\labels", exist_ok=True)
os.makedirs(f"{output_folder}\\valid\\images", exist_ok=True)
os.makedirs(f"{output_folder}\\valid\\labels", exist_ok=True)



def txt_generator(image_name, file_text):
    with open(image_name.replace('.jpg', '.txt'), "a") as file:
        file.write(file_text) 

def move_image(file_path, outputpah):
    shutil.copy(file_path, outputpah)

with open(f"{main_folder}\\{csv_file}", 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    data_list = list(csv_reader)

    for row in data_list:
        try:
            img_location = f"{main_folder}\\{image_folder}\\{row[0]}"
            image =  cv2.imread(img_location)
            if row[1] != "0":
                mask = rle_to_mask(row[2], image.shape[:2])
                annotation = draw_square_around_mask(image, mask, row[1])
            else:
                annotation = ""  
            if random.random() < 0.1:  # Вероятность 0.1 для попадания в valid
                destination_folder = f"{output_folder}\\valid"
            else:
                destination_folder = f"{output_folder}\\train"
     
            txt_generator(destination_folder + f"\\labels\\{row[0]}", annotation)
            move_image(img_location, destination_folder + "\\images")
        except Exception as e:
            print(f"Файл {row[0]}.", f"Ошибка {e}")
        #break
    print("Программа завершена.")
