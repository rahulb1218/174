import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

dir_with_people = "./data/validation/with_person"
dir_without_people = "./data/validation/without_person"

def detect_person_by_color(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin_percentage = (np.sum(mask > 0) / mask.size) * 100
    return skin_percentage > 5 

def detect_person_by_shape(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_height, max_height = 100, 400
    min_width, max_width = 30, 200
    min_aspect_ratio, max_aspect_ratio = 2, 8

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = height / float(width)
        if (min_height <= height <= max_height and
            min_width <= width <= max_width and
            min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            return True
    return False

def process_images(directory, label):
    correct_detections = 0
    total_images = 0
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        
        is_person_present_by_color = detect_person_by_color(frame)
        is_person_present_by_shape = detect_person_by_shape(frame)
        is_person_present = is_person_present_by_color and is_person_present_by_shape
        
        if (is_person_present and label == "With Person") or (not is_person_present and label == "Without Person"):
            correct_detections += 1
        total_images += 1
    
    if total_images > 0:
        accuracy = (correct_detections / total_images) * 100
    else:
        accuracy = 0
    return accuracy, total_images

accuracy_with, total_with = process_images(dir_with_people, "With Person")
accuracy_without, total_without = process_images(dir_without_people, "Without Person")

total_images = total_with + total_without
overall_accuracy = ((accuracy_with * total_with) + (accuracy_without * total_without)) / total_images

print(f"Accuracy for images with people: {accuracy_with:.2f}%")
print(f"Accuracy for images without people: {accuracy_without:.2f}%")
print(f"Overall accuracy: {overall_accuracy:.2f}%")

categories = ['With People', 'Without People', 'Overall']
values = [accuracy_with, accuracy_without, overall_accuracy]
plt.figure(figsize=(8, 4))
plt.bar(categories, values, color=['blue', 'green', 'red'])
plt.xlabel('Categories')
plt.ylabel('Accuracy (%)')
plt.title('Detection Accuracy')
plt.ylim(0, 100)
plt.show()