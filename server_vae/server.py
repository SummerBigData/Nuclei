from flask import Flask, url_for, send_file
from flask.json import jsonify
import cv2 as cv
from scipy.misc import imread
from os.path import join, isfile

app = Flask(__name__)

win_name = 'Image'
base_path = '/n/home00/lerner.67/nuclei/test_data2'

@app.route('/')
def index():
    return app.send_static_file('vae_tmp.html')

@app.route('/disp/<id>')
def disp(id):
    id = str(id)
    path = join(base_path, id, 'images', id+'.png')
    if not isfile(path):
        return None, 500 


    return send_file(path, mimetype='image/png'), 200
    #img = imread(path)
    #img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

    #json = {'data': img.tolist()}
    #return jsonify(json), 200

if __name__ == '__main__':
    app.run(port=8080)
