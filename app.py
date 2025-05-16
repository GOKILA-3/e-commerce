import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model("fashion_mnist_advanced_model.h5")

# Labels for Fashion MNIST
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load overlay images
specs_img = cv2.imread("specs.png", cv2.IMREAD_UNCHANGED)
cap_img = cv2.imread("cap.png", cv2.IMREAD_UNCHANGED)

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def overlay_transparent(background, overlay, x, y, scale=1.0):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape

    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha_overlay * overlay[:, :, c] +
            alpha_background * background[y:y+h, x:x+w, c]
        )
    return background

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Face overlays (specs & cap)
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            # Overlay specs
            specs_scale = box_w / specs_img.shape[1] * 0.8
            frame = overlay_transparent(frame, specs_img, x, y + int(box_h * 0.4), scale=specs_scale)

            # Overlay cap
            cap_scale = box_w / cap_img.shape[1] * 1.1
            frame = overlay_transparent(frame, cap_img, x - 10, y - int(box_h * 0.8), scale=cap_scale)

    # Model Prediction (optional snapshot)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype("float32") / 255.0
    input_img = np.expand_dims(normalized, axis=(0, -1))  # shape: (1, 28, 28, 1)

    pred = model.predict(input_img, verbose=0)
    class_id = np.argmax(pred)
    pred_label = labels[class_id]

    # Show prediction
    cv2.putText(frame, f"Prediction: {pred_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI Virtual Try-On", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
