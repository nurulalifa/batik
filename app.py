import re
from flask import Flask, render_template, request
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import cv2
import numpy as np

from flask_cors import CORS, cross_origin

names = ["Batik A", "Batik B", "Batik C", "Batik D", "Batik"]


# Process image and predict label
def processImg(IMG_PATH):
    # Read image
    model = load_model("modelbatik.h5")
    
    # Preprocess image
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (200, 200))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    res = model.predict(image)
    label = np.argmax(res)
    print("Label", label)
    labelName = names[label]
    print("Label name:", labelName)

    return render_template ('index2.html', labelName = names[label])


# Initializing flask application
app = Flask(__name__)
cors = CORS(app)


# About page with render template
@app.route("/")
def postsPage():
    return render_template("index.html")

# Process images
@app.route("/", methods=["POST"])
def processReq():
    data = request.files["file"]
    data.save("img.jpg")
    resp = processImg("img.jpg")
    return resp



if __name__ == "__main__":
    app.run(port = 8000, debug=True)