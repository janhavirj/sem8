import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist  # type: ignore

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Limit data to speed up for testing
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

def preprocess_image(img):
    """Resize & normalize image."""
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0
    return img

# Preprocess images
X_train_preprocessed = [preprocess_image(img) for img in X_train]
X_test_preprocessed = [preprocess_image(img) for img in X_test]

def extract_hog_features(img):
    """Extract HOG features from image."""
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

# Extract features
X_train_features = np.array([extract_hog_features(img) for img in X_train_preprocessed])
X_test_features = np.array([extract_hog_features(img) for img in X_test_preprocessed])

# Scale features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

# Train model
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train_features, y_train)

# Predict
y_pred = clf.predict(X_test_features)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Select one image per digit (0-9)
selected_indices = []
for digit in range(10):
    idx = np.where(y_test == digit)[0][0]
    selected_indices.append(idx)

# Print predictions for these 10 digits
print("\nPredicted digits for samples 0-9:")
for i, idx in enumerate(selected_indices):
    print(f"Digit {i}: Predicted = {y_pred[idx]}")

# Plot images
plt.figure(figsize=(12, 6))
for i, idx in enumerate(selected_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


