import cv2
import os

# Set the number of images to capture for each sign
num_images = 100

# Path where the images will be saved
dataset_path = 'data/'

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is open
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    print("Press 'q' to quit.")
    # Create the folder for the current letter if it doesn't exist
    letter = input("Enter the letter you want to capture images for (A-Z): ").upper()
    letter_path = os.path.join(dataset_path, letter)

    if not os.path.exists(letter_path):
        os.makedirs(letter_path)
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the image
        cv2.imshow(f"Capturing {letter} - {count+1}/{num_images}", frame)

        # Save the image
        img_name = os.path.join(letter_path, f"{letter}_{count+1}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1

        # Press 'q' to quit after capturing images
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()