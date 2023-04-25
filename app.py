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


app = Flask(__name__)

creds = Credentials.from_authorized_user_file('C:\JS_Projects\Final_Project_Camera\client_secret_credential.json', scopes=['https://www.googleapis.com/auth/drive'])
notebook_id = '100izQVIYwv0nJJ5hMdmyFuQ6Li3sEy9T'

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

# @app.route('/')
# def home():
#     return render_template('upload.html')

@app.route('/',methods=['GET', 'POST'])
def test():
    return render_template('test.html')

@app.route('/result')
def result():
    return render_template('result.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_image():
#     if request.method == 'POST':
#         if 'image' in request.files:
#             # The image was uploaded
#             image = request.files['image']
#             # Do something with the uploaded image
#             return 'Image uploaded successfully'
#         else:
#             # The image was captured
#             image = request.data
#             # Do something with the captured image
#             return 'Image captured successfully'
#     return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        return redirect(url_for('extract_text', filename=filename))
    return render_template('upload.html')


@app.route('/process', methods=['POST'])
def process_image():
    image_file = request.files['file']
    text_output = get_text_from_image(image_file)
    return render_template('result.html', text_output=text_output)


@app.route('/extract_text', methods=['POST'])
def extract_text():
    try:
        # Authenticate and access the notebook using its ID
        notebook_id = '100izQVIYwv0nJJ5hMdmyFuQ6Li3sEy9T'
        creds = Credentials.from_authorized_user_file('C:\JS_Projects\Final_Project_Camera\client_secret_credential.json', scopes=['https://www.googleapis.com/auth/drive'])
        drive_service = build('drive', 'v3', credentials=creds)
        file = drive_service.files().get(fileId='100izQVIYwv0nJJ5hMdmyFuQ6Li3sEy9T').execute()
        notebook_title = file['name']

        # Get the uploaded image file
        image = request.files['file']

        # Set up the request data
        files = {
            'file': (image.filename, image.stream, 'image/jpeg'),
            'notebook_title': 'ML_NOTEBOOK'
        }

        # Send the image to the notebook for text extraction
        url = 'https://colab.research.google.com/drive/{}?usp=sharing'.format('100izQVIYwv0nJJ5hMdmyFuQ6Li3sEy9T')
        response = requests.post(url, files=files)

        # Check if the response was successful
        if response.status_code == requests.codes.ok:
            # Return the extracted text
            return response.text
        else:
            # Return an error message
            return 'An error occurred while processing the image'
    except HttpError as error:
        # Return the error message
        return f"An error occurred: {error}"



def get_notebook_file():
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        file = drive_service.files().get(fileId='100izQVIYwv0nJJ5hMdmyFuQ6Li3sEy9T').execute()
        print(f"Notebook title: {file['name']}")
        return file
    except HttpError as error:
        print(f"An error occurred: {error}")

def get_text_from_image(image_file):
    # Authenticate and access notebook
    creds = Credentials.from_authorized_user_file('C:\JS_Projects\Final_Project_Camera\client_secret_credential.json', scopes=['https://www.googleapis.com/auth/drive'])
    drive_service = build('drive', 'v3', credentials=creds)
    
    # Upload image file to notebook
    url = 'https://colab.research.google.com/drive/{}?usp=sharing'.format('100izQVIYwv0nJJ5hMdmyFuQ6Li3sEy9T')
    files = {'file': image_file}
    response = requests.post(url, files=files)
    print(response.status_code)
    
    # Retrieve text output from notebook
    file_id = drive_service.files().list(q=f"name='{'100izQVIYwv0nJJ5hMdmyFuQ6Li3sEy9T'}'").execute()['files'][0]['id']
    text_output = drive_service.files().export(fileId=file_id, mimeType='text/plain').execute()
    return text_output.decode('utf-8')







if __name__ == '__main__':
    app.run(debug=True)