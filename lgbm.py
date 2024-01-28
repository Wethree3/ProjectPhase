import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os

# Define the paths to your image folders
data_dir = 'data/'

# Set the image dimensions and the number of classes
img_width, img_height = 224, 224
num_classes = 8

# Set up data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load a pre-trained MobileNetV2 model for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Extract features and labels
features = []
labels = []

for i in range(generator.samples // generator.batch_size):
    x, y = generator.next()
    features.extend(base_model.predict(x))
    labels.extend(np.argmax(y, axis=1))

X = np.array(features)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the features to be compatible with XGBoost
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Create an XGBoost classifier
model = xgb.XGBClassifier(objective="multi:softmax", num_class=num_classes)

# Train the XGBoost model
model.fit(X_train_flattened, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_flattened)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")