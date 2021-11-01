from flask import Flask, redirect, url_for, render_template, request, flash , jsonify
import time
import os
from skimage import io
import cv2
from my_tools.train import train
from my_tools.histogrameFeatures import HistogrameFeatures
from my_tools.search import Search

UPLOAD_FOLDER = 'static/images'
hsv_init = (8,12,3)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('main.html' )

@app.post('/upload')
def upload():
    file = request.files['image']
    new_file_name = str(
        str(time.time()) + '.png'
    )
    file.save(os.path.join(
            app.config['UPLOAD_FOLDER'],new_file_name
        )
    )
    train()

    histFeatures = HistogrameFeatures(hsv_init)

    query = cv2.imread(os.path.join(os.path.dirname(__file__), 'static/images/' + new_file_name))
    features = histFeatures.features(query)
    #searching
    searcher = Search('./index.csv')
    results = searcher.search(features)
    RESULTS_LIST = list()
    for (score, pathImage) in results:
        RESULTS_LIST.append(
            {"image": str(pathImage), "score": str(score)}
        )

    return jsonify(RESULTS_LIST)






if __name__ == '__main__':
    app.run(debug=True)