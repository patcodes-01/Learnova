# ocr/ocr.py

import io
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------
# IMAGE OCR (JPG / PNG)
# ---------------------------------------

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='eng')
    return text

# ---------------------------------------
# PDF → OCR
# ---------------------------------------

def extract_text_from_pdf(pdf_path, use_snider=False, snider_key=None):
    pages = convert_from_path(pdf_path)
    all_text = []

    for pil_page in pages:
        if use_snider and snider_key:
            # Snider AI API case (if you add later)
            buf = io.BytesIO()
            pil_page.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            text = ocr_snider_api(img_bytes, api_key=snider_key)
        else:
            text = pytesseract.image_to_string(pil_page, lang="eng")

        all_text.append(text)

    return "\n".join(all_text)

# ---------------------------------------
# Optional Snider API placeholder
# ---------------------------------------

def ocr_snider_api(image_bytes, api_key=None):
    raise NotImplementedError("Add Snider API integration here if needed.")

