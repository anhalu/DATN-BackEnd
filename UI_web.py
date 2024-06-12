from typing import Union

import PIL.Image
import cv2
import gradio as gr
import numpy as np
import requests
import json
from loguru import logger

headers = {
    'accept': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxNzI0NDI2ODQ2fQ.crJ4mm0iHKkfSbQ8bI3qjzdax4Jsi_o29eiEzC2Cclo'
}


def process(image: Union[np.ndarray, PIL.Image.Image]):
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # upload api
    url = "http://0.0.0.0:8000/api/v1/file/upload"
    _, img_encoded = cv2.imencode('.jpg', image)
    image_bytes = img_encoded.tobytes()

    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
    response = requests.request("POST", url, headers=headers, files=files)
    logger.info(f"Response code: {response.json()}")
    if response.status_code == 200:
        file_id = response.json()["file_id"]
        logger.info(f"File id : {file_id}")
        # run ocr general

        url = f"http://0.0.0.0:8000/api/v1/ocr/general?file_id={file_id}&export_file=false&cached=true"

        response = requests.request("POST", url, headers=headers)

        if response.status_code == 200:
            # get filed
            url = "http://0.0.0.0:8000/api/v1/ocr/parse-with-group-rule"

            payload = json.load(open('request_api.json'))
            payload['request_id'] = file_id

            headers_new = {
                'accept': 'application/json',
                'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxNzI0NDg1NDE5fQ.bwXl5fZWgg_YbgVy9cLzfrU4L9KHDrCWEX2LJWzfd7A',
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers_new, json=payload)
            if response.status_code == 200:
                results = ""
                results_json = response.json()[0]
                logger.info(results_json)
                for field in results_json["fields"]:
                    if len(field["values"]) > 0 and field["name"] != 'title':
                        results += f'{field["name"]}: \t {field["values"][0]["value"]} \n'
                return results
            else:
                return "Run Rule Based Failed !"
        else:
            return "Run ocr general failed !"
    else:
        return "Failed to send image to server !"


demo = gr.Interface(
    title="Demo Extract Information From Ielts Cerf",
    fn=process,
    inputs=gr.Image(),
    outputs=['text'],
)
demo.launch()
