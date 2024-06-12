import cv2

def yolo_to_rectangle(annotation, img_width, img_height):
    class_id, center_x, center_y, width, height = map(float, annotation.strip().split())
    
    center_x = center_x * img_width
    center_y = center_y * img_height
    width = width * img_width
    height = height * img_height
    

    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return class_id, x1, y1, x2, y2

def annotate_image(yolo_annotation_path, img_path):

    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    with open(yolo_annotation_path, 'r') as file:
        annotations = file.readlines()

    for annotation in annotations:
        class_id, x1, y1, x2, y2 = yolo_to_rectangle(annotation, img_width, img_height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Annotated Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


file_name = "044700866" #Название файла аннотации и изображения
output_dir_images = "output/train/images/"
output_dir_labels = "output/train/labels/"

yolo_annotation_path = f"{output_dir_labels}{file_name}.txt"
img_path = f"{output_dir_images}{file_name}.jpg"

annotate_image(yolo_annotation_path, img_path)


