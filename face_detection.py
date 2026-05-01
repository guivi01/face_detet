import argparse
import cv2
import sys


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError("Could not load Haar cascade face detector.")

    return face_detector


def detect_faces(frame, face_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces


def draw_faces(frame, faces):
    for x, y, w, h in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return frame


def run_on_image(image_path):
    face_detector = load_face_detector()

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)

    faces = detect_faces(image, face_detector)
    output = draw_faces(image, faces)

    print(f"Detected {len(faces)} face(s).")

    cv2.imshow("Face Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_webcam(camera_index=0):
    face_detector = load_face_detector()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        faces = detect_faces(frame, face_detector)
        output = draw_faces(frame, faces)

        cv2.putText(
            output,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Simple face detection with OpenCV")

    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image file. If omitted, webcam mode is used."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for webcam mode. Default is 0."
    )

    args = parser.parse_args()

    if args.image:
        run_on_image(args.image)
    else:
        run_on_webcam(args.camera)


if __name__ == "__main__":
    main()