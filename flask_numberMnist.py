# -*- coding: utf-8 -*-


from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# pip install opencv-python
app = Flask(__name__)
name='numberMnist web'

static='static/'
model=load_model("model/numberMnist.keras")

def prediction(filename):
    img = cv2.imread(filename)
    x=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x=x.reshape(1, 28*28)/255
    results=model.predict(x)
    result=np.argmax(results, axis=1)
    return result[0]
    
    


# http://localhost:5000/upload
@app.route('/upload', methods=['GET','POST'])
def upload_file():    
    if request.method != 'POST':
        return render_template('index.html')
    else :  #post form에서 이미지 업로드        
        file = request.files['upload']
        filename = file.filename
        file.save(static+filename)
        result=prediction(static+filename)
        print(filename, 'upload======')
        return render_template('index.html', 
                               image_name=filename, result=result)


if __name__ == '__main__':
    app.run(debug=True, port="5000")
    
    
