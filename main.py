import os
import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from asanas import asanas_name

# Load the saved model
model = tf.keras.models.load_model("yoga_pose_model.h5")

# Function to preprocess the image and predict the pose
def predict_pose(image_path, model):
    img = cv2.imread(image_path, 1)  # Read the image
    resized_img = cv2.resize(img, (100, 100)) / 255.0  # Resize and normalize
    input_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(input_img)

    # Convert prediction to pose label
    predicted_pose_label = np.argmax(prediction)

    return predicted_pose_label

# Function to handle image selection and pose prediction
def handle_image_selection():
    image_path = filedialog.askopenfilename(
        title="Choose an Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )

    if image_path:
        # Display the selected image in the UI
        img = Image.open(image_path)
        img = img.convert("RGB")
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Predict the pose
        predicted_pose_label = predict_pose(image_path, model)

        # Get the pose name
        predicted_asana = asanas_name.get(predicted_pose_label, "Unknown Pose")

        # Display the result
        result_text.set(f"Predicted Pose: {predicted_asana}")
    else:
        result_text.set("No image selected.")

# Set up the GUI
root = tk.Tk()
root.title("Yoga Pose Detection")

# Instructions label
instruction_label = Label(
    root, text="Click 'Select Image' to choose a yoga pose image for prediction.", font=("Helvetica", 12)
)
instruction_label.pack(pady=10)

# Image display area
image_label = Label(root)
image_label.pack(pady=10)

# Result display
result_text = tk.StringVar()
result_label = Label(root, textvariable=result_text, font=("Helvetica", 12, "bold"))
result_label.pack(pady=10)

# Button to select an image
select_button = Button(root, text="Select Image", command=handle_image_selection, font=("Helvetica", 12))
select_button.pack(pady=20)

# Start the GUI loop
root.mainloop()
