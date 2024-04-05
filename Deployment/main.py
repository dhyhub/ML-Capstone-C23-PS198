import io
import os

from PIL import Image
from flask import Flask, request
from marshmallow import Schema, fields, ValidationError

from google.cloud import storage
from ultralytics import YOLO

BUCKET_NAME = "nutriscan-c23-ps198-assets"

STORAGE_CLIENT = storage.Client(project="nutriscan-c23-ps198")
BUCKET         = STORAGE_CLIENT.bucket(BUCKET_NAME)


app   = Flask(__name__)
model = YOLO('model.pt')


# ? Retrieve image from Google Cloud Bucket and return it as bytes object
def get_image_from_bucket(image_file):
    """
    Retrieve image from Google Cloud Bucket and return it as bytes object
    @param image_file: URL of the image in the bucket
    @return: bytes object of the image
    """
    img_blob = BUCKET.blob(image_file)
    contents = img_blob.download_as_bytes()

    return contents


# ? Schema for the request body
class RequestSchema(Schema):
    photo = fields.Str(required=True)



@app.route("/detect", methods=["POST"])
def detect():
    # ? Validate the request method
    if request.method != "POST":
        return {"message": "Invalid request method"}, 400
    
    # ? Validate the request body
    request_body = request.json
    schema       = RequestSchema()
    
    print(request_body)
    
    try:
        body = schema.load(request_body)
    except ValidationError as err:
        return {"message": err.messages}, 400
        
    # ? Get the file from Google Cloud Bucket
    # * body["photo"]: URL of the image in the bucket
    image_bytes = get_image_from_bucket(body.get("photo"))
    img         = Image.open(io.BytesIO(image_bytes))
    
    # ? Predict the image
    result        = model.predict(img)
    result_labels = [
        model.names[label_num]
        for label_num
        in result[0].boxes.cls.tolist()
    ]
    
    # ? Return the result
    return {"labels": result_labels}, 200


# ngrok_tunnel = ngrok.connect(8000)
# print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
