from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

home_view = Blueprint('home_view', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@home_view.route('/')  # Route for the page
def display_home_page():
	return 'This is the index page!'

# route to upload image via POST using 'formdata'
# image is not yet saved
@home_view.route('/sendImage', methods = ['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return "No file selected"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("Image successfully received!")
        print(file)
        return jsonify(message="Successfully received image!")