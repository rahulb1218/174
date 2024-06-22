import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

random.seed(42)

model = load_model('model_human_detection_sgd.keras')

optimal_threshold = 0.5

def predict_and_show(image_path):
    img = load_img(image_path, target_size=(256, 256))
    plt.imshow(img)
    plt.show()
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0][0]
    print("Prediction score:", prediction)
    classified_as = 'with_person' if prediction < optimal_threshold else 'without_person'
    print("Classified as:", classified_as)
    return classified_as

dir_path_without_person = '/Users/vandit/Library/Mobile Documents/com~apple~CloudDocs/Documents/174ProjectItr2/data/validation/without_person'
dir_path_with_person = '/Users/vandit/Library/Mobile Documents/com~apple~CloudDocs/Documents/174ProjectItr2/data/validation/with_person'

all_images_without_person = [os.path.join(dir_path_without_person, f) for f in os.listdir(dir_path_without_person) if os.path.isfile(os.path.join(dir_path_without_person, f))]
all_images_with_person = [os.path.join(dir_path_with_person, f) for f in os.listdir(dir_path_with_person) if os.path.isfile(os.path.join(dir_path_with_person, f))]

selected_images_without_person = random.sample(all_images_without_person, min(1000, len(all_images_without_person)))
selected_images_with_person = random.sample(all_images_with_person, min(1000, len(all_images_with_person)))

count_correct_without = 0
count_correct_with = 0

for image in selected_images_without_person:
    if predict_and_show(image) == 'without_person':
        count_correct_without += 1

for image in selected_images_with_person:
    if predict_and_show(image) == 'with_person':
        count_correct_with += 1

average_accuracy_without = count_correct_without / len(selected_images_without_person)
average_accuracy_with = count_correct_with / len(selected_images_with_person)

print("Average accuracy for without person:", average_accuracy_without)
print("Average accuracy for with person:", average_accuracy_with)

categories = ['Without Person', 'With Person']
values = [average_accuracy_without, average_accuracy_with]
plt.figure(figsize=(8, 4))
plt.bar(categories, values, color=['red', 'green'])
plt.xlabel('Categories')
plt.ylabel('Accuracy (%)')
plt.title('Detection Accuracy')
plt.ylim(0, 1)
plt.show()
