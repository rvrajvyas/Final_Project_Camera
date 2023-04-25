import google.oauth2.credentials
import requests
import os
import base64
import io
import flask_uploads
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from flask import Flask, render_template, request , redirect, url_for 
from flask_uploads import UploadSet, IMAGES, configure_uploads

from werkzeug.utils import secure_filename

#Importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter
import tensorflow as tf
import os
import glob
import imutils

UPLOAD_FOLDER = 'C:\\JS_Projects\\Final_Project_Camera\\uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def make_ocr():
    return

@app.route('/',methods=['GET', 'POST'])
def test():
    return render_template('result.html')

@app.route('/result',methods=['GET','POST'])
def ocr_result():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread(UPLOAD_FOLDER+'/'+filename)
        ocr_img = make_ocr(img)
        ocr_img_name = filename.split('.')[0]+"_ocr.jpg"
        _ = cv2.imwrite(UPLOAD_FOLDER+'/'+ocr_img_name, ocr_img)
        return render_template('result.html',org_img_name=filename,ocr_img_name=ocr_img_name)


# @app.route('/result')
# def result():
#     return render_template('result.html')


# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST' and 'photo' in request.files:
#         filename = photos.save(request.files['photo'])
#         return redirect(url_for('extract_text', filename=filename))
#     return render_template('upload.html')




if __name__ == '__main__':
    app.run(debug=True)