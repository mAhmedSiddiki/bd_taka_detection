#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import bd_taka_processing

#load model
model =load_model("model/taka_model.h5")

print('@@ Model loaded')

def pred_taka(taka):
  test_image = load_img(taka, target_size = (250, 120))
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  
  result = model.predict(test_image).round(3)
  print('@@ Raw result = ', result)

  pred = np.argmax(result)

  if pred == 0:
    return "1 Taka", bd_taka_processing.taka_1(taka)
  elif pred == 1:
    return '10 Taka', bd_taka_processing.taka_10(taka)
  elif pred == 2:
    return '100 Taka', bd_taka_processing.taka_100(taka)
  elif pred == 3:
    return '1000 Taka', bd_taka_processing.taka_1000(taka)
  elif pred == 4:
    return '2 Taka', bd_taka_processing.taka_2(taka)
  elif pred == 5:
    return "20 Taka", bd_taka_processing.taka_20(taka)
  elif pred == 6:
    return '200 Taka', bd_taka_processing.taka_200(taka)
  elif pred == 7:
    return '5 Taka', bd_taka_processing.taka_5(taka)
  elif pred == 8:
    return '50 Taka', bd_taka_processing.taka_50(taka)
  elif pred == 9:
    return '500 Taka', bd_taka_processing.taka_500(taka)
  




# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, ori_fake = pred_taka(taka=file_path)
        
        return render_template('predict_taka.html', pred_output = pred, user_image = file_path, ori_fake = ori_fake)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
    