import cv2
from datetime import datetime

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to save an image.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Live Webcam Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved image as {filename}")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()