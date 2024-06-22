import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

test_dir = '/Users/vandit/Library/Mobile Documents/com~apple~CloudDocs/Documents/174ProjectAvnoor/testingData'

img_height = 128
img_width = 128
batch_size = 16

model = tf.keras.models.load_model('model.keras')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  
)

predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = (predictions > 0.5).astype(int)

true_labels = test_generator.classes

cm = confusion_matrix(true_labels, predicted_classes)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Without Person', 'With Person'], yticklabels=['Without Person', 'With Person'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

report = classification_report(true_labels, predicted_classes, target_names=['Without Person', 'With Person'])
print(report)

accuracy_without_person = accuracy_score(true_labels[true_labels == 0], predicted_classes[true_labels == 0])
accuracy_with_person = accuracy_score(true_labels[true_labels == 1], predicted_classes[true_labels == 1])

categories = ['Without Person', 'With Person']
accuracies = [accuracy_without_person, accuracy_with_person]

plt.figure(figsize=(10, 5))
plt.bar(categories, accuracies, color=['red', 'green'])
plt.ylim(0, 1)
plt.xlabel('Categories')
plt.ylabel('Accuracy (%)')
plt.title('Detection Accuracy')
plt.show()