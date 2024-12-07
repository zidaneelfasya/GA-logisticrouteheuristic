from PIL import Image
import pytesseract

# Load the uploaded image
image_path = "image.png"
image = Image.open(image_path)

# Extract text from the image using OCR
extracted_text = pytesseract.image_to_string(image)
extracted_text
