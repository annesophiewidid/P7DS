# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:17:59 2021

@author: glass
"""

from flask import Flask
from flask_restful import Resource, Api
import os
import pandas as pd
import joblib


app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def get(self, client_id):
        # PATH = os.getcwd()
        # PATH += "\\"
        # f = PATH+'application_train.csv'

        df = pd.read_csv('application_train.csv',
                 low_memory=False,
                 verbose=False,
                 encoding='UTF-8',
                 dtype={'Special': 'object'}
                 )
        X = df.drop(columns=['TARGET'])
        # y = df['TARGET']

        #chargement du preprocessor
        loaded_preprocessor = joblib.load('preprocessor.pkl')
    
        #chargement du modèle
        loaded_model = joblib.load('model.pkl')

        data_clientunique = X[X['SK_ID_CURR']==str(client_id)]
     
        data_clientunique=loaded_preprocessor.transform(data_clientunique)
        score_client=loaded_model.predict_proba(data_clientunique)
        value = score_client[0][1]*100
        return value

api.add_resource(Prediction, '/prediction/<int:client_id>/')

if __name__ == '__main__':

    app.run(debug=True)

