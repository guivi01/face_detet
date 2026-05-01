import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MODEL_DIR = "models"
PROTOTXT_FILE = "deploy.prototxt"
CAFFEMODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/3.4.0/"
    "samples/dnn/face_detector/deploy.prototxt"
)

CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


def download_file(url, output_path):
    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 0:
        return

    print(f"Downloading: {output_path.name}")
    urllib.request.urlretrieve(url, output_path)


def ensure_model_files(model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    prototxt_path = model_dir / PROTOTXT_FILE
    caffemodel_path = model_dir / CAFFEMODEL_FILE

    download_file(PROTOTXT_URL, prototxt_path)
    download_file(CAFFEMODEL_URL, caffemodel_path)

    if not prototxt_path.exists():
        raise FileNotFoundError(f"Missing model config: {prototxt_path}")

    if not caffemodel_path.exists():
        raise FileNotFoundError(f"Missing model weights: {caffemodel_path}")

    if caffemodel_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"The Caffe model file looks too small: {caffemodel_path}\n"
            "Delete it and rerun the script."
        )

    return prototxt_path, caffemodel_path


def load_dnn_face_detector(model_dir):
    prototxt_path, caffemodel_path = ensure_model_files(model_dir)

    net = cv2.dnn.readNetFromCaffe(
        str(prototxt_path),
        str(caffemodel_path)
    )

    if net.empty():
        raise RuntimeError("Could not load OpenCV DNN face detector.")

    return net


def find_image_files(folder):
    return [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def detect_faces_dnn(image, net, confidence_threshold=0.55):
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < confidence_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array(
            [width, height, width, height]
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:
            continue

        faces.append({
            "box": (x1, y1, x2, y2),
            "confidence": confidence,
            "area": face_width * face_height
        })

    return faces


def crop_face_with_padding(image, box, padding=0.25, output_size=None):
    height, width = image.shape[:2]

    x1, y1, x2, y2 = box

    face_width = x2 - x1
    face_height = y2 - y1

    pad = int(max(face_width, face_height) * padding)

    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(width, x2 + pad)
    crop_y2 = min(height, y2 + pad)

    face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if face_crop.size == 0:
        return None

    if output_size is not None:
        face_crop = cv2.resize(face_crop, output_size)

    return face_crop


def prepare_dataset(
    input_dir,
    output_dir,
    model_dir,
    confidence_threshold,
    padding,
    output_size,
    save_all_faces
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    print(f"Current working directory: {Path.cwd()}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {confidence_threshold}")
    print()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    class_dirs = [
        path for path in sorted(input_dir.iterdir())
        if path.is_dir()
    ]

    if not class_dirs:
        direct_images = find_image_files(input_dir)

        if direct_images:
            raise RuntimeError(
                "Images were found directly inside raw_dataset, but they need "
                "to be inside class folders.\n\n"
                "Use this structure:\n"
                "raw_dataset/\n"
                "  me/\n"
                "    image1.jpg\n"
                "  unknown/\n"
                "    image1.jpg\n"
            )

        raise RuntimeError(
            f"No class folders found inside: {input_dir}\n\n"
            "Expected structure:\n"
            "raw_dataset/\n"
            "  class_1/\n"
            "    image1.jpg\n"
            "  class_2/\n"
            "    image1.jpg\n"
        )

    net = load_dnn_face_detector(model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_images_found = 0
    total_saved = 0
    total_no_face = 0
    total_unreadable = 0

    print("Classes found:")

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = find_image_files(class_dir)

        print(f"  {class_name}: {len(image_files)} image(s)")

        if not image_files:
            continue

        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_files:
            total_images_found += 1

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Unreadable image: {image_path}")
                total_unreadable += 1
                continue

            faces = detect_faces_dnn(
                image,
                net,
                confidence_threshold=confidence_threshold
            )

            if not faces:
                print(f"No face detected: {image_path}")
                total_no_face += 1
                continue

            faces = sorted(
                faces,
                key=lambda face: face["area"],
                reverse=True
            )

            if not save_all_faces:
                faces = faces[:1]

            for face_index, face in enumerate(faces):
                face_crop = crop_face_with_padding(
                    image=image,
                    box=face["box"],
                    padding=padding,
                    output_size=output_size
                )

                if face_crop is None:
                    print(f"Could not crop face: {image_path}")
                    total_unreadable += 1
                    continue

                if save_all_faces:
                    output_filename = (
                        f"{image_path.stem}_face_{face_index + 1}.jpg"
                    )
                else:
                    output_filename = f"{image_path.stem}_face.jpg"

                output_path = output_class_dir / output_filename

                success = cv2.imwrite(str(output_path), face_crop)

                if not success:
                    print(f"Could not save face crop: {output_path}")
                    total_unreadable += 1
                    continue

                total_saved += 1

    print()
    print("Done.")
    print(f"Images found: {total_images_found}")
    print(f"Saved face crops: {total_saved}")
    print(f"No face detected: {total_no_face}")
    print(f"Unreadable or failed images: {total_unreadable}")

    if total_images_found == 0:
        raise RuntimeError(
            "No images were found inside your class folders. "
            "Check your folder names and image extensions."
        )

    if total_saved == 0:
        raise RuntimeError(
            "Images were found, but no faces were cropped.\n\n"
            "Try lowering the confidence threshold:\n"
            "python prepare_faces.py --confidence 0.35\n"
        )


def parse_output_size(value):
    if value is None:
        return None

    value = value.lower().strip()

    if value in {"none", "original", "no"}:
        return None

    if "x" not in value:
        raise argparse.ArgumentTypeError(
            "Output size must look like 160x160 or use 'none'."
        )

    width, height = value.split("x", 1)

    return int(width), int(height)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cropped face dataset using OpenCV DNN CNN face detector."
    )

    parser.add_argument(
        "--input",
        default="raw_dataset",
        help="Input folder containing class folders. Default: raw_dataset"
    )

    parser.add_argument(
        "--output",
        default="dataset_faces",
        help="Output folder for cropped faces. Default: dataset_faces"
    )

    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Folder where DNN model files are stored. Default: models"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Minimum face detection confidence. Default: 0.55"
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Padding around detected face. Default: 0.25"
    )

    parser.add_argument(
        "--output-size",
        type=parse_output_size,
        default="160x160",
        help="Resize cropped faces, for example 160x160. Use 'none' to keep original crop size."
    )

    parser.add_argument(
        "--save-all-faces",
        action="store_true",
        help="Save every detected face instead of only the largest face."
    )

    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        model_dir=args.model_dir,
        confidence_threshold=args.confidence,
        padding=args.padding,
        output_size=args.output_size,
        save_all_faces=args.save_all_faces
    )


if __name__ == "__main__":
    main()