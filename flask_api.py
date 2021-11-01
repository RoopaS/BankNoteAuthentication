# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:50:11 2021

@author: Roopa
"""

from flask import Flask, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
pickle_in = open('clf1.pkl','rb')
clf1 = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome Roopa retry'

@app.route('/predict')
def pred_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy') 
    pred = clf1.predict([[variance,skewness,curtosis,entropy]])
    return 'Predicted value: '+ str(pred)

@app.route('/predict_file',methods=["POST"])
def pred_note_authen_file():
    df_test = pd.read_csv(request.files.get("file"))
    pred = clf1.predict(df_test)
    return 'Predicted value for csv test file: '+ str(list(pred))


if __name__ == '__main__':
    app.run()