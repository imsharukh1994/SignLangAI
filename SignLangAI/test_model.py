import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model/ASL_model.h5')

# Class labels (A-Z)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is open
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame to match model input size
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image

    # Make predictions
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_letter = class_labels[predicted_class[0]]

    # Display the prediction on the frame
    cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("ASL Sign Language Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
