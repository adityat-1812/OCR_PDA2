import cv2
import numpy as np
import pytesseract
from PIL import Image
from tensorflow.keras.models import load_model

# Load the custom CNN model
model = load_model('CNN.h5')
# Load the NLP model
nlp_model = load_model('NLP.h5')

# Set the path to Tesseract executable (change it according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Preprocessing parameters
rescale_factor = 1.5
threshold_value = 180

# Load and preprocess the image
image_path = '../IMG/test-02.jpg'
image = cv2.imread(image_path)
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