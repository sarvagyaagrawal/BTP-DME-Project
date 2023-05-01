from flask_cors import CORS, cross_origin
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
import requests
import json
from final_script import *

app = Flask(__name__)


cors = CORS(app) 
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/get_annotation" ,methods = ['POST'])
@cross_origin()
def main():
    image_base64=request.form['image_base64']
    return jsonify(get_annotations(image_base64))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)