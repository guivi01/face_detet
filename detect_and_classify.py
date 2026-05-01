import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "face_classifier.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (160, 160)


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade face detector.")

    return detector


def load_classifier():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_classifier.py first."
        )

    if not Path(CLASS_NAMES_PATH).exists():
        raise FileNotFoundError(
            f"{CLASS_NAMES_PATH} not found. Run train_classifier.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, IMG_SIZE)

    array = np.expand_dims(face_rgb, axis=0)
    return array


def classify_face(face_bgr, model, class_names):
    array = preprocess_face(face_bgr)

    predictions = model.predict(array, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])

    label = class_names[class_index]

    return label, confidence


def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


def run_webcam(camera_index, confidence_threshold):
    detector = load_face_detector()
    model, class_names = load_classifier()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        faces = detect_faces(frame, detector)

        for x, y, w, h in faces:
            face_crop = frame[y:y + h, x:x + w]

            label, confidence = classify_face(face_crop, model, class_names)

            if confidence < confidence_threshold:
                display_text = f"unknown {confidence:.2f}"
            else:
                display_text = f"{label} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                display_text,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Detection + Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detect and classify faces from webcam."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default is 0."
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="Minimum confidence before showing predicted class."
    )

    args = parser.parse_args()

    run_webcam(args.camera, args.confidence)


if __name__ == "__main__":
    main()