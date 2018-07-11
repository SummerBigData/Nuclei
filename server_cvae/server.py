from flask import Flask, url_for, send_file
from flask.json import jsonify
from os.path import join, isfile

app = Flask(__name__)
base_path = '/n/home00/lerner.67/nuclei'

@app.route('/')
def index():
    return app.send_static_file('cvae_tmp.html')

@app.route('/disp/<dir>/<id>')
def disp(dir, id):
    dir = str(dir)
    id = str(id)
    print dir, id
    path = join(base_path, dir, id, 'images', id+'.png')
    if not isfile(path):
        return None, 500 

    return send_file(path, mimetype='image/png'), 200

if __name__ == '__main__':
    app.run(port=8080)
