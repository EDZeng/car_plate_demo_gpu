from hubconf import Model, select_device, custom
import numpy as np
from PIL import Image
import torch
import cv2
import re
from paddleocr import PaddleOCR, draw_ocr
from datetime import datetime


def extract_tuples(data):           
    tuples = []
    if isinstance(data, tuple):
        tuples.append(data)
    elif isinstance(data, list):
        for item in data:
            tuples.extend(extract_tuples(item))
    return tuples


def process_image_and_extract_text(image_path, model, ocr, output_file):
    results_plate = model(image_path)
    original_image = cv2.imread(image_path)
    
    for det in results_plate.pred[0]:
        words_to_filter = []
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_region = original_image[y1:y2, x1:x2]
        extracted_text = extract_text_from_image(cropped_region, ocr)
        if extracted_text:
            words_to_filter.append(extracted_text)
            print(extracted_text)
            
            pattern = r'^(?!IPC)(?=.*[^\d])(?=.*[a-zA-Z]).{4,7}$'      
            with open(output_file, 'a', encoding='utf-8') as file:
                for word in words_to_filter:
                    if re.match(pattern, word):
                        file.write(word + '\n')
                        return word
    results_plate.print()
    results_plate.save()


def extract_text_from_image(image, ocr):
    results = ocr.ocr(image)
    all_tuples = extract_tuples(results)
    if all_tuples:
        final_result = all_tuples[0][0]
        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', final_result)        
        return cleaned_text
    return None


if __name__ == "__main__":
    model_plate = custom(path_or_model='best.pt')
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    today_date = datetime.now().strftime('%Y%m%d')
    output_file = f"captures/{today_date}/output.txt"
    img_path = '../test10.jpg'
    process_image_and_extract_text(img_path, model_plate, ocr, output_file)

















