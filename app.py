import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load your trained model
finished_model_train = tf.keras.models.load_model('bestoneyet.h5')

# Labels of your classes
labels = ['B3', 'Kaca', 'Kertas', 'Metal', 'Organik', 'Plastik', 'Residu']

# Function to predict the class of the image
def predict_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(512, 512))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = finished_model_train.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[predicted_class_index]
    probabilities = "\n".join([f"{label}: {prob:.2f}" for label, prob in zip(labels, predictions[0])])

    return predicted_label, probabilities

# Function to open file dialog and display image
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    display_and_predict(file_path)

# Function to capture an image from the webcam
def capture_image():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Show the webcam feed
        cv2.imshow("Capture Image (Press SPACE to take photo, ESC to exit)", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == 32:  # SPACE key to capture
            # Save the captured image
            file_path = "captured_image.jpg"
            cv2.imwrite(file_path, frame)
            display_and_predict(file_path)
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to display the image and predict its class
def display_and_predict(file_path):
    # Display the image
    img = Image.open(file_path)
    img.thumbnail((300, 300))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Make a prediction
    predicted_label, probabilities = predict_image(file_path)
    result_label.config(text=f"Predicted Class: {predicted_label}")
    probabilities_label.config(text=f"Probabilities:\n{probabilities}")

# Create the main Tkinter window
app = tk.Tk()
app.title("Image Classification App")

# Add a label to display the image
img_label = Label(app)
img_label.pack()

# Add a button to open a file dialog
upload_button = Button(app, text="Upload Image", command=open_file)
upload_button.pack()

# Add a button to capture an image with the webcam
capture_button = Button(app, text="Capture Image", command=capture_image)
capture_button.pack()

# Add a label to display the prediction result
result_label = Label(app, text="Predicted Class: ", font=("Helvetica", 14))
result_label.pack()

# Add a label to display the prediction probabilities
probabilities_label = Label(app, text="", font=("Helvetica", 10), justify="left")
probabilities_label.pack()

# Run the Tkinter event loop
app.mainloop()
