from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import os
import cv2 as cv
import numpy as np
app = Flask(__name__)

dic = {0: 'Bacterial', 1: 'Healthy'}

model = load_model('model.h5')

model.make_predict_function()


def predict_label(img_path):
    im_path = img_path
    img = cv.resize(cv.imread(im_path), (244, 244))
    x = np.expand_dims(img, axis=0)
    result = model.predict(x)
    print(result)
    if (result[0][0] < result[0][1]):
        return f"Peach leaf is a Bacterial Spot Peach Leaf with {result[0][1]}% assurity"
    else:
        return f"Peach leaf is a Healthy Peach Leaf with {result[0][0]}% assurity"



# routes
@app.route("/", methods=['GET', 'POST'])
def box():
    if request.method == 'GET':
        return render_template('index.html')



@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files.get('image')

        img_path = "static/predict.JPG" 
        img.save(img_path)
        
        p = predict_label(img_path)
        #os.remove(img_path)

    return p


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
