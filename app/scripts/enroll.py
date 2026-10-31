import os
import cv2
import numpy as np
from ultralytics import YOLO
from app.config import Settings
from app.core.face import FaceProcessor


def main():
    settings = Settings()
    person_name = input('Person Name: ').strip()

    if not person_name:
        print("Person name cannot be empty.")
        return

    # Create directories for body and face images
    body_dir = settings.gallery_dir / person_name / 'images'
    face_dir = settings.gallery_dir / person_name / 'faces'
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    max_images = 50
    conf_threshold = 0.7
    min_width, min_height = 80, 160
    blur_threshold = 100

    # Load YOLO segmentation model
    model = YOLO(str(settings.detector_model))

    # Load face processor
    face_processor = FaceProcessor(
        model_name=settings.face_model,
        det_size=settings.face_det_size,
    )

    # Open the saved video
    video_path = r"E:\reid_tracker_v01-main\SampleData\WIN_20261026_22_52_38_Pro.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Video could not be opened. Check the video path.")
        cap.release()
        return

    count = len([
        f for f in body_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.png')
    ])

    if count >= max_images:
        print('Enrollment already completed.')
        cap.release()
        return

    save_pressed = False
    print('\nSPACE : Save image\nESC   : Exit\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        result = model(frame, verbose=False)[0]
        body_crop = None
        face_crop = None

        if result.masks is not None:
            best_idx = -1
            best_area = 0

            for i, box in enumerate(result.boxes):
                if int(box.cls.item()) != 0 or float(box.conf.item()) < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area > best_area:
                    best_area = area
                    best_idx = i

            if best_idx != -1:
                mask = result.masks.data[best_idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, result.boxes[best_idx].xyxy[0])

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) >= min_width and (y2 - y1) >= min_height:
                    mask = cv2.resize(
                        mask, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    binary = (mask > 0.5).astype(np.uint8)
                    segmented = frame.copy()
                    segmented[binary == 0] = 0
                    body_crop = segmented[y1:y2, x1:x2]

                    if body_crop.size != 0:
                        body_crop = cv2.resize(body_crop, (128, 256))
                        gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
                        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

                        if blur < blur_threshold:
                            body_crop = None
                        else:
                            person_roi = frame[y1:y2, x1:x2]

                            if person_roi.size != 0:
                                _, face_emb = face_processor.extract(person_roi)

                                if face_emb is not None:
                                    faces = face_processor.app.get(person_roi)

                                    if faces:
                                        face = max(
                                            faces,
                                            key=lambda f: (
                                                (f.bbox[2] - f.bbox[0])
                                                * (f.bbox[3] - f.bbox[1])
                                            )
                                        )

                                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                                        fx1 = max(0, fx1)
                                        fy1 = max(0, fy1)
                                        fx2 = min(person_roi.shape[1], fx2)
                                        fy2 = min(person_roi.shape[0], fy2)

                                        if fx2 > fx1 and fy2 > fy1:
                                            face_crop = person_roi[fy1:fy2, fx1:fx2]

                                            if face_crop.size != 0:
                                                face_crop = cv2.resize(
                                                    face_crop, (112, 112)
                                                )
                                                cv2.imshow('Face', face_crop)

                            cv2.rectangle(
                                display, (x1, y1), (x2, y2), (0, 255, 0), 2
                            )
                            cv2.putText(
                                display, f'{count}/{max_images}',
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                            )
                            cv2.imshow('Person', body_crop)

        cv2.imshow('Enrollment', display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == ord(' '):
            if not save_pressed and body_crop is not None:
                body_path = body_dir / f'{count:04d}.jpg'
                saved = cv2.imwrite(str(body_path), body_crop)

                if saved:
                    print(f'Saved body {count:04d}.jpg')

                    if face_crop is not None:
                        face_path = face_dir / f'{count:04d}.jpg'
                        if cv2.imwrite(str(face_path), face_crop):
                            print(f'Saved face {count:04d}.jpg')

                    count += 1
                    save_pressed = True

                    if count >= max_images:
                        break
            else:
                print("No clear person detected. Try another video frame.")
        else:
            save_pressed = False

    cap.release()
    cv2.destroyAllWindows()
    print(f'Enrollment completed with {count} body images.')


if __name__ == '__main__':
    main()