from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_sslify import SSLify

from utils import Recognize
app = Flask(__name__)
socketio = SocketIO(app)
sslify = SSLify(app)



@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('image')
def catch_image(image):
    names = Recognize(image)
    print(names)
    emit('result', names)

if __name__ == '__main__':
    socketio.run(app, debug=False, host='192.168.0.119', port=5000)