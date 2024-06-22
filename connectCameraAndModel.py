import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

model = tf.keras.models.load_model('model.keras')

def preprocess_image(frame):
    image = cv2.resize(frame, (256, 256)) 
    
    # plt.imshow(image)
    # plt.show()
    
    image_array = img_to_array(image) # new
    
    image_array = np.expand_dims(image_array, axis=0) 
    
    
    image_array /= 255.0 # new
    return image_array

threshold = .5

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image = preprocess_image(frame)
        
        predictions = model.predict(image)
        predicted_probability = predictions[0][0]  
        classification_label = 'With Person' if predicted_probability < threshold else 'Without Person'
        if classification_label == 'With Person':
            print(f'Frame classified as: {classification_label} with probability {predicted_probability:.4f}')
            cv2.putText(frame, "Person detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        

        cv2.imshow('Webcam', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()