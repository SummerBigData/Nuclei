from flask import Flask, url_for, send_file
from flask.json import jsonify
from os.path import join, isfile

app = Flask(__name__)
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

if __name__ == '__main__':
    app.run(port=8080)
