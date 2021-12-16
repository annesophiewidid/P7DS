# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:17:59 2021

@author: glass
"""

from flask import Flask
from flask_restful import Resource, Api, abort
import os
import pandas as pd
import joblib
import json

app = Flask(__name__)
api = Api(app)

PATH = os.getcwd()
PATH += "\\"
f = PATH+'application_train.csv'

df = pd.read_csv(f,
                 low_memory=False,
                 verbose=False,
                 encoding='ISO-8859-1',
                 dtype={'Special': 'object'}
                 )
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# CLIENTS= {X}

#chargement du preprocessor
loaded_preprocessor = joblib.load('preprocessor.pkl')
    
#chargement du modèle
loaded_model = joblib.load('model.pkl')

data_clientunique = X[X['SK_ID_CURR']==100003]
     
data_clientunique=loaded_preprocessor.transform(data_clientunique)

class Prediction(Resource):
    def get(self, client_id):
        score_client=loaded_model.predict_proba(data_clientunique)
        response = json.dumps({'response': score_client})
        return response, 200

api.add_resource(Prediction, '/prediction/<int:client_id>/')

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/predict', methods=['GET'])
# def predict():
#     # parse input features from request
#     request_json = request.get_json()
#     x = float(request_json['input'])
 
#     # load model
#     model = load_models()
#     prediction = model.predict([[x]])[0]
#     response = json.dumps({'response': prediction})
#     return response, 200
# if __name__ == '__main__':
#     application.run(debug=True)