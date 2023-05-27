import os
import re
from google.cloud import vision
import tkinter as tk
from tkinter import filedialog
import sys
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import pytesseract
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

def CNN_NLP(image_path):
    # Load the custom CNN model
    model = load_model('CNN.h5')
    # Load the NLP model
    nlp_model = load_model('NLP.h5')

    # Preprocessing parameters
    rescale_factor = 1.5
    threshold_value = 180

    image = Image.open(image_path)

    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rescaled = cv2.resize(gray, None, fx=rescale_factor, fy=rescale_factor, interpolation=cv2.INTER_CUBIC)
    _, threshold = cv2.threshold(rescaled, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours and extract text regions
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter out small regions
            text_regions.append((x, y, w, h))

    # Perform character recognition on each text region
    recognized_text = ''
    for region in text_regions:
        (x, y, w, h) = region
        region_image = threshold[y:y+h, x:x+w]

        # Resize region image to match the input size of the CNN model
        region_image = cv2.resize(region_image, (28, 28))

        # Preprocess region image if needed
        # Add your own preprocessing steps here (e.g., normalization)

        # Reshape and normalize image for the CNN model
        region_image = np.reshape(region_image, (1, 28, 28, 1))
        region_image = region_image / 255.0

        # Predict character using the CNN model
        predicted_class = np.argmax(model.predict(region_image), axis=-1)
        predicted_char = chr(predicted_class[0] + 65)  # Assuming the model predicts characters in the range A-Z

        # Perform NLP prediction using the NLP model
        predicted_nlp = nlp_model.predict(region_image)  # Adjust input as per NLP model requirements

        recognized_text += predicted_char + ' ' + predicted_nlp

    # Use Tesseract for additional OCR
    additional_text = pytesseract.image_to_string(Image.fromarray(threshold), config='--psm 6')

    # Aggregate the recognized text
    final_text = recognized_text + ' ' + additional_text
    print(final_text)

def preprocess_image(image_path):
    """Preprocesses the image to enhance text visibility."""
    image = Image.open(image_path)

    # Convert to grayscale
    image = image.convert("L")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Apply adaptive thresholding
    img_array = np.array(image)
    _, thresholded_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Denoise the image
    denoised_img = cv2.fastNlMeansDenoising(thresholded_img, None, 10, 7, 21)

    # Convert back to PIL Image
    preprocessed_image = Image.fromarray(denoised_img)

    return preprocessed_image

def detect_prescription(image_path):
    """Detects text in an image using Google Vision API."""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vis.json'
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Set language hints to detect only English text
    image_context = vision.ImageContext(language_hints=["en", "mr", "mul_digits"])

    response = client.text_detection(image=image, image_context=image_context)
    texts = response.text_annotations

    text_data = ""
    for text in texts:
        text_data += text.description + "\n"

    return text_data

def detect_text_with_tesseract(image_path):
    """Detects text in an image using Tesseract OCR."""
    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

    # Extract numbers using regular expressions
    numbers = re.findall(r'\d+', text)
    extracted_numbers = ' '.join(numbers)

    return extracted_numbers

def save_text_to_file(text, file_path):
    
    # Calculate the index to start cutting from
    cut_index = int(len(selected_text) * 0.7)

    # Cut the text from the calculated index to the end
    cut_text = text[''':cut_index''']

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(cut_text)

    # Remove the temporary image file
    os.remove(temp_image_path)
    os.remove(image_path)
    
    sys.exit()

def display_text_in_window(text, save_file_path):
    root = tk.Tk()
    root.title("Text Detection")

    def save_and_exit():
        save_text_to_file(text_widget.get("1.0", tk.END), save_file_path)
        root.destroy()

    text_widget = tk.Text(root)
    text_widget.pack()

    # Split text into lines
    lines = text.split('\n')

    # Select only half of the lines
    selected_lines = lines[:len(lines) // 2]

    # Join the selected lines back into a single string with preserved indentation
    selected_text = '\n'.join(selected_lines)

    text_widget.insert(tk.END, selected_text)

    save_button = tk.Button(root, text="Save", command=save_and_exit)
    save_button.pack(side=tk.BOTTOM)

    # Set the window to be always on top
    root.attributes("-topmost", True)

    root.mainloop()

if __name__ == '__main__':
    # Example usage
    image_path = 'prescriptions/temp_images/output.jpg'
    save_file_path = 'prescriptions/text_files/input.txt'

    # Detect text using CNN and NLP models pre-builded
    CNN_NLP(image_path)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Save the preprocessed image as a temporary file
    temp_image_path = 'prescriptions/temp_images/ot.jpg'
    preprocessed_image.save(temp_image_path)

    # Use Google Vision API for text detection
    extracted_text = detect_prescription(image_path)

    # If no text is detected, fall back to Tesseract OCR
    if not extracted_text:
        extracted_text = detect_text_with_tesseract(image_path)

    display_text_in_window(extracted_text, save_file_path)
