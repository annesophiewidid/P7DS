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

CLIENTS= {X}

#chargement du preprocessor
loaded_preprocessor = joblib.load('preprocessor.pkl')
    
#chargement du modèle
loaded_model = joblib.load('model.pkl')


client_id = X[X['SK_ID_CURR']==int(id_input)]
client_id=loaded_preprocessor.transform(client_id)
score_client=loaded_model.predict_proba(client_id)

class Prediction(Resource):
    def get(self, client_id):   
        return "score_client"

api.add_resource(Prediction, '/prediction/<int:client_id>/')

if __name__ == '__main__':
    app.run(debug=True)

