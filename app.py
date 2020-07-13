from flask import Flask
from flask import render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import base64
import skimage.io
from svm import SVM
from extract_feature import extract_feature_hog
app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('home.html', y_pred = None)

@socketio.on('image', namespace = '/test')
def get_image(json):
    data = json['data'].split(';')[1].split(',')[1]
    data = base64.decodebytes(data.encode('utf-8'))
    img = skimage.io.imread(data, plugin = 'imageio', as_gray = True)
    feature = extract_feature_hog(img)
    y_pred = SVM.predict(feature)
    emit('result', {'data': str(y_pred)})
if __name__ == "__main__":
    SVM.load_model()
    app.run(debug=True)
