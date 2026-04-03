import cv2
import os
import numpy as np

def apply_mask_to_face(frame, mask, x, y, w, h):
    mask_x = x - int(0.10 * w)
    mask_y = y - int(0.15 * h)
    mask_w = int(1.20 * w)
    mask_h = int(1.25 * h)

    if mask_x < 0:
        mask_x = 0
    if mask_y < 0:
        mask_y = 0
    if mask_x + mask_w > frame.shape[1]:
        mask_w = frame.shape[1] - mask_x
    if mask_y + mask_h > frame.shape[0]:
        mask_h = frame.shape[0] - mask_y

    if mask_w <= 0 or mask_h <= 0:
        return frame

    mask_resized = cv2.resize(mask, (mask_w, mask_h))

    if mask_resized.shape[2] == 4:
        mask_rgb = mask_resized[:, :, :3]
        mask_alpha = mask_resized[:, :, 3] / 255.0

        roi = frame[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w]
        alpha = np.dstack((mask_alpha, mask_alpha, mask_alpha))

        blended = (mask_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        frame[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w] = blended

    return frame


cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)

mask = cv2.imread("cartoon_mask_yellow.png", cv2.IMREAD_UNCHANGED)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        frame = apply_mask_to_face(frame, mask, x, y, w, h)

    cv2.imshow("Face Mask Video", frame)

    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break

    if cv2.getWindowProperty("Face Mask Video", cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()