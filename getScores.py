import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('model_human_detection_sgd.keras')

def predict_images(image_paths):
    prediction_scores = []
    count = 0
    for image_path in image_paths:
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        count += 1
        print("count: ", count) 
        print("Prediction score:", prediction[0][0])
        prediction_scores.append(prediction[0][0])
    
    average_prediction_score = np.mean(prediction_scores)
    print("Average Prediction Score for without:", average_prediction_score)

dir_path = '/Users/vandit/Library/Mobile Documents/com~apple~CloudDocs/Documents/174ProjectItr2/data/validation/without_person'

all_images = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

selected_images = random.sample(all_images, 500)

predict_images(selected_images)