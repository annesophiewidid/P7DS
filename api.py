# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1>Pret à dépenser</h1><p>This site is a prototype API for PRET A DEPENSER.</p>"

app.run()


