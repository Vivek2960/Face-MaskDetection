# USAGE
# python model.py --dataset dataset

import os
import numpy as np
from imutils import paths
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import cv2
print(cv2.__version__)
from sklearn.metrics import roc_curve


dataset_path = r"C:\Users\91635\PycharmProjects\Face_MaskDetection\dataset1"
plot_path = "plot.png"
model_path = "mask_detector.model"

INIT_LR = 1e-4
EPOCHS = 2
BS = 32

print(dataset_path)
subfolders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

if os.path.exists(dataset_path):
    subfolders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
else:
    print(f"Error: Dataset path '{dataset_path}' does not exist.")

print("[INFO] Loading images...")
image_paths = list(paths.list_images(dataset_path))
data = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

train_X, test_X, train_Y, test_Y = train_test_split(data, labels,
                                                    test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

base_model = MobileNetV2(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable = False

print("[INFO] Compiling model...")
# Update optimizer creation to use learning_rate instead of lr
optimizer = Adam(learning_rate=INIT_LR)

model.compile(loss="binary_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

print("[INFO] Training head...")
history = model.fit(
    aug.flow(train_X, train_Y, batch_size=BS),
    steps_per_epoch=len(train_X) // BS,
    validation_data=(test_X, test_Y),
    validation_steps=len(test_X) // BS,
    epochs=EPOCHS
)

print("[INFO] Evaluating network...")
pred_probs = model.predict(test_X, batch_size=BS)
pred_classes = np.argmax(pred_probs, axis=1)

print(classification_report(test_Y.argmax(axis=1), pred_classes,
                            target_names=lb.classes_))
print(f"Shape of test_Y: {test_Y.shape}")
print(f"Shape of pred_probs: {pred_probs.shape}")
# If test_Y is one-hot encoded, convert it to a 1D array of class labels
test_Y = test_Y.argmax(axis=1)

# Ensure test_Y is not ravelled or flattened
# test_Y should be a 1D array of true labels (not one-hot encoded)
# pred_probs should be a 2D array with probabilities for each class

# Compute the ROC curve for the positive class (assuming binary classification)
fpr, tpr, _ = roc_curve(test_Y, pred_probs[:, 1])

# Now, you can proceed with plotting or further analysis
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_Y.ravel(), pred_probs[:, 1].ravel())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")

print("[INFO] Saving mask detector model...")
# Assuming you want to save in the recommended Keras format
model_path = "mask_detector_model.keras"

# Save the model using the Keras format
model.save(model_path)


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_path)
