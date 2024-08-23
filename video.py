from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, face_net, mask_net):
    # Grab the dimensions of the frame and construct a blob from it
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Initialize lists for faces, their locations, and predictions from the face mask network
    faces = []
    locations = []
    predictions = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is above the minimum confidence threshold
        if confidence > args["confidence"]:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(width - 1, end_x), min(height - 1, end_y))

            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Add the face and bounding box locations to their respective lists
            faces.append(face)
            locations.append((start_x, start_y, end_x, end_y))

    # Perform predictions if at least one face was detected
    if len(faces) > 0:
        # Perform batch predictions on all faces at once instead of one-by-one predictions
        faces = np.array(faces, dtype="float32")
        predictions = mask_net.predict(faces, batch_size=32)

    # Return a tuple of face locations and their corresponding predictions
    return (locations, predictions)


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxt_path = r"C:\Users\91635\PycharmProjects\Face_MaskDetection\face_detector\deploy.prototxt"
weights_path = r"C:\Users\91635\PycharmProjects\Face_MaskDetection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

# Load the face mask detector model from disk
print("[INFO] Loading face mask detector model...")
mask_net = load_model("mask_detector_model.keras")

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to a maximum width of 400 pixels
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)

    # Detect faces in the frame and determine if they are wearing a face mask or not
    (locations, predictions) = detect_and_predict_mask(frame, face_net, mask_net)

    # Loop over the detected face locations and their corresponding predictions
    for (box, prediction) in zip(locations, predictions):
        # Unpack the bounding box and predictions
        (start_x, start_y, end_x, end_y) = box
        (mask, without_mask) = prediction

        # Determine the class label and color to draw the bounding box and text
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
video_stream.stop()

