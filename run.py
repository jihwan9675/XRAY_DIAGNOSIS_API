# app.py
from flask import Flask, render_template
from flask import Flask, Blueprint, request, render_template, flash, redirect, url_for
from flask import current_app as app
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import csv
import cv2
import sys
import numpy
import h5py
import time
import skimage.draw
import datetime
import socket
import hashlib
import multiprocessing
import pydicom
import numpy as np

#Flask 객체 인스턴스 생성
app = Flask(__name__)


@app.route('/')  # 접속하는 url
def index():
  return "Hi"


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save("./static/uploader/original/"+request.files['file'].filename)
        print("Saved File : "+"./static/uploader/" +
              request.files['file'].filename)

        return 'http://192.168.40.48/static/uploader/heatmap/'+request.files['file'].filename

    elif request.method == 'GET':
        return 'hi'


def ImagePreProcessing():
    pass


def Predict_Xray():
    pass


if __name__ == "__main__":
  app.run(host="0.0.0.0", port="80", debug=True)
