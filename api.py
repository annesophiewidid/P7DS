# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:17:59 2021

@author: glass
"""

from flask import Flask
from flask_restful import Resource, Api, abort

app = Flask(__name__)
api = Api(app)

CLIENTS= {}

def abort_if_client_doesnt_exist(client_id):
    if client_id not in CLIENTS:
        abort(404, message="Client {} doesn't exist".format(client_id))

# sélectionner un client

    def get(self, client_id):
        abort_if_client_doesnt_exist(client_id)
        return CLIENTS[client_id]

    def delete(self, client_id):
        abort_if_client_doesnt_exist(client_id)
        del CLIENTS[client_id]
        return '', 204

if __name__ == '__main__':
    app.run(debug=True)

