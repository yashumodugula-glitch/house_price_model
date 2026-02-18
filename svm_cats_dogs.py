# ==========================================
# TASK 03 - SVM Cats vs Dogs
# ==========================================

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset path (do NOT change unless needed)
DATASET_PATH = "dataset/train"

IMG_SIZE = 64
LIMIT_IMAGES = 200  # limit for faster training

X = []
y = []

print("Loading images...")

for category in ["cats", "dogs"]:
    path = os.path.join(DATASET_PATH, category)
    label = 0 if category == "cats" else 1

    if not os.path.exists(path):
        print("❌ Folder not found:", path)
        exit()

    count = 0
    for img in os.listdir(path):
        if count >= LIMIT_IMAGES:
            break

        try:
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            X.append(image.flatten())
            y.append(label)
            count += 1
        except:
            continue

X = np.array(X)
y = np.array(y)

print("Total images loaded:", len(X))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM model...")

model = SVC(kernel="linear")
model.fit(X_train, y_train)

print("Testing model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\n✅ Task 03 Completed Successfully!")
