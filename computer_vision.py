import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from skimage.feature import hog
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import albumentations as A

# Paths
DATASET_PATH = r"C:\\cif_editor\\corel10k_10"
PROCESSED_DATA_PATH = r"C:\\cif_editor\\processed_images10"

# Load images and labels
def load_images_and_labels(folder_path):
    X, y = [], []
    class_names = sorted(os.listdir(folder_path))
    label_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith('.jpg'):
                    file_path = os.path.join(class_path, file)
                    image = cv2.imread(file_path)
                    if image is not None:
                        X.append(cv2.resize(image, (125, 185)))
                        y.append(label_mapping[class_name])

    return np.array(X), np.array(y), label_mapping

# Preprocess images
def preprocess_image(image):
    return cv2.medianBlur(image, 1)

# Segment images using GMM
def segment_image_gmm(image):
    pixels = image.reshape(-1, 3) / 255.0
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(pixels)
    segmented = gmm.predict(pixels).reshape(image.shape[:2])

    # Apply Gaussian smoothing to the segmented image
    smoothed = gaussian_filter(segmented.astype(float), sigma=1)
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
        mser.setDelta(3)
        mser.setMinArea(30)
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

        kaze = cv2.KAZE_create(
            extended=True,
            threshold=0.005
        )
        keypoints, descriptors = kaze.detectAndCompute(image, None)
        if descriptors is not None:
            return np.mean(descriptors, axis=0)
        else:
            return np.zeros((64,), dtype=np.float32)
    except Exception as e:
        print(f"Error extracting KAZE features: {e}")
        return None

# Data augmentation
def augment_image(image):
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5),
    ])
    augmented = augmentations(image=image)
    return augmented['image']

# Augment dataset
def augment_dataset(X, y, augment_times=3):
    augmented_images = []
    augmented_labels = []

    for image, label in zip(X, y):
        # Add original image
        augmented_images.append(cv2.resize(image, (125, 185)))
        augmented_labels.append(label)

        # Augmentation
        for _ in range(augment_times):
            augmented_image = augment_image(image)
            augmented_image = cv2.resize(augmented_image, (125, 185))
            augmented_images.append(augmented_image)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

# Prepare dataset with augmentation
def prepare_dataset_with_augmentation(augment_times=2):
    X, y, label_mapping = load_images_and_labels(DATASET_PATH)

    # Split into training and testing data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Augment training data
    X_train, y_train = augment_dataset(X_train, y_train, augment_times)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    return X_train, X_test, y_train, y_test

# Train MLP model
def train_mlp_model(X_train, y_train, X_test, y_test):
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
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

    class_names = sorted(os.listdir(DATASET_PATH))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

# Main function
def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Path {DATASET_PATH} does not exist!")
        return

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Use dataset with augmentation
    X_train, X_test, y_train, y_test = prepare_dataset_with_augmentation(augment_times=2)
    if X_train.size == 0 or y_train.size == 0:
        print("Dataset preparation failed. Check your data directory.")
        return

    train_mlp_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
