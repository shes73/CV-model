import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from skimage.feature import hog
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Paths
TRAIN_PATH = r"C:\cif_editor\corel10k_train"
TEST_PATH = r"C:\cif_editor\corel10k_test"
PROCESSED_DATA_PATH = r"C:\cif_editor\processed_images10"

# Load images and labels
def load_images_from_folder(folder_path):
    class_data = {}
    class_names = sorted(os.listdir(folder_path))

    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            images = []
            for file in os.listdir(class_path):
                if file.lower().endswith('.jpg'):
                    file_path = os.path.join(class_path, file)
                    image = cv2.imread(file_path)
                    if image is not None:
                        images.append(image)
            class_data[class_name] = images

    return class_data

def prepare_manual_dataset():
    # Load training and testing data
    train_data = load_images_from_folder(TRAIN_PATH)
    test_data = load_images_from_folder(TEST_PATH)

    # Combine into features and labels
    def process_data(class_data):
        X, y = [], []
        label_mapping = {class_name: idx for idx, class_name in enumerate(class_data.keys())}
        for class_name, images in class_data.items():
            labels = [label_mapping[class_name]] * len(images)
            X.extend([cv2.resize(img, (125, 185)) for img in images])
            y.extend(labels)
        return np.array(X), np.array(y)

    X_train, y_train = process_data(train_data)
    X_test, y_test = process_data(test_data)

    return X_train, X_test, y_train, y_test

# Preprocess images
def preprocess_image(image):
    return cv2.medianBlur(image, 1)

# Segment images using GMM
def segment_image_gmm(image):
    pixels = image.reshape(-1, 3) / 255.0
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(pixels)
    segmented = gmm.predict(pixels).reshape(image.shape[:2])
    smoothed = median_filter(segmented, size=3)
    return smoothed

# Extract HOG features
def extract_hog_features(image):
    try:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        fd = hog(
            image, orientations=12, pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), visualize=False, channel_axis=None
        )
        return fd
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        return None

# Extract MSER features
def extract_mser_features(image):
    try:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(image)
        return np.array([len(regions)], dtype=np.float32)
    except Exception as e:
        print(f"Error extracting MSER features: {e}")
        return None

# Extract KAZE features
def extract_kaze_features(image):
    try:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kaze = cv2.KAZE_create()
        keypoints, descriptors = kaze.detectAndCompute(image, None)
        if descriptors is not None:
            return np.mean(descriptors, axis=0)
        else:
            return np.zeros((64,), dtype=np.float32)
    except Exception as e:
        print(f"Error extracting KAZE features: {e}")
        return None

# Save processed images
def save_processed_image(image, idx, label):
    class_name = sorted(os.listdir(TRAIN_PATH))[label]
    class_path = os.path.join(PROCESSED_DATA_PATH, class_name)
    os.makedirs(class_path, exist_ok=True)
    file_path = os.path.join(class_path, f"processed_{idx}.png")
    cv2.imwrite(file_path, image)

# Prepare dataset with combined features
def prepare_dataset():
    X_train, X_test, y_train, y_test = prepare_manual_dataset()
    features_train, features_test = [], []
    labels_train, labels_test = [], []

    # Gather data by class
    for idx, (image, label) in enumerate(zip(X_train, y_train)):
        processed_image = preprocess_image(image)
        segmented_image = segment_image_gmm(processed_image)

        smoothed_image = cv2.normalize(segmented_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        smoothed_image = cv2.resize(smoothed_image, (125, 185))

        save_processed_image(smoothed_image, idx, label)

        hog_features = extract_hog_features(smoothed_image)
        mser_features = extract_mser_features(smoothed_image)
        kaze_features = extract_kaze_features(smoothed_image)

        if hog_features is None or mser_features is None or kaze_features is None:
            continue

        combined_features = np.hstack((hog_features, mser_features, kaze_features))
        features_train.append(combined_features)
        labels_train.append(label)

    for idx, (image, label) in enumerate(zip(X_test, y_test)):
        processed_image = preprocess_image(image)
        segmented_image = segment_image_gmm(processed_image)

        smoothed_image = cv2.normalize(segmented_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        smoothed_image = cv2.resize(smoothed_image, (125, 185))

        hog_features = extract_hog_features(smoothed_image)
        mser_features = extract_mser_features(smoothed_image)
        kaze_features = extract_kaze_features(smoothed_image)

        if hog_features is None or mser_features is None or kaze_features is None:
            continue

        combined_features = np.hstack((hog_features, mser_features, kaze_features))
        features_test.append(combined_features)
        labels_test.append(label)

    # Convert lists to arrays
    X_train = np.array(features_train, dtype=np.float32)
    y_train = np.array(labels_train, dtype=np.int32)
    X_test = np.array(features_test, dtype=np.float32)
    y_test = np.array(labels_test, dtype=np.int32)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Train MLP model
def train_mlp_model(X_train, y_train, X_test, y_test):
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    class_names = sorted(os.listdir(TRAIN_PATH))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

# Main function
def main():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print(f"Paths {TRAIN_PATH} or {TEST_PATH} do not exist!")
        return

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_dataset()
    if X_train.size == 0 or y_train.size == 0:
        print("Dataset preparation failed. Check your data directory.")
        return

    train_mlp_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
