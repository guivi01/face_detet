import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade face detector.")

    return detector


def crop_largest_face(image, detector, padding=0.25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    pad = int(max(w, h) * padding)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    return image[y1:y2, x1:x2]


def prepare_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    detector = load_face_detector()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_skipped = 0

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            path for path in class_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_files:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                total_skipped += 1
                continue

            face_crop = crop_largest_face(image, detector)

            if face_crop is None:
                print(f"No face found: {image_path}")
                total_skipped += 1
                continue

            output_path = output_class_dir / f"{image_path.stem}_face.jpg"
            cv2.imwrite(str(output_path), face_crop)
            total_saved += 1

    print(f"Done.")
    print(f"Saved face crops: {total_saved}")
    print(f"Skipped images: {total_skipped}")


def main():
    parser = argparse.ArgumentParser(description="Prepare cropped face dataset.")
    parser.add_argument("--input", default="raw_dataset", help="Raw dataset folder")
    parser.add_argument("--output", default="dataset_faces", help="Output cropped face folder")

    args = parser.parse_args()

    prepare_dataset(args.input, args.output)


if __name__ == "__main__":
    main()