# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:02:43 2021

@author: glass
"""

import os
import csv
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_curve
from sklearn import metrics
import numpy as np
import time
from urllib.request import urlopen
import json


st.write("""
# "Prêt à dépenser" Prediction App
Cette application prédit l'octroi du crédit à un client en fonction de son profil!
""")
st.write('---')

# Loads the pret a depenser Dataset
PATH = os.getcwd()
PATH += "\\"
f = PATH+'application_train.csv'

#with open(f, 'r') as csvfile:  # Sniffing csv to find separator
#    dialect = csv.Sniffer().sniff(csvfile.readline())
#    sep = dialect.delimiter

df = pd.read_csv(f,
                 low_memory=False,
                 verbose=False,
                 encoding='ISO-8859-1',
                 dtype={'Special': 'object'}
                 )

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_explanation(id_input, dataframe, model, sample):
    return interpretation(str(id_input), 
        dataframe, 
        model, 
        sample=sample)

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_explanation(id_input, dataframe, model, sample):
    return interpretation(str(id_input), 
        dataframe, 
        model, 
        sample=sample)

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_ligne_data(id, df):
    return df[df['SK_ID_CURR']==int(id)].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

liste_id = df['SK_ID_CURR'].tolist()

df_pay = df[df.TARGET == 0]
df_unpay = df[df.TARGET == 1]
X = df.drop(columns=['TARGET'])
#for col in X.columns:
#    if col.endswith("_MEDI") or col.endswith("_MODE"):
#        X.drop(columns=col, inplace=True)

y = df['TARGET']


#affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
#chaine = "l'id Saisi est " + str(id_input)
#st.write(chaine)

sample_en_regle = str(list(df[df['TARGET'] == 0].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples d\'id de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(df[df['TARGET'] == 1].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_defaut)
    st.write(chaine_en_regle)

elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API

    #Appel de l'API : 

    API_url = "http://127.0.0.1:5000/credit/" + id_input

    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)

        API_data = json.loads(json_url.read())
        classe_predite = API_data['prediction']
        if classe_predite == 1:
            etat = 'client à risque'
        else:
            etat = 'client peu risqué'
        proba = 1-API_data['proba'] 

        #affichage de la prédiction
        prediction = API_data['proba']
        classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
        classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'

    st.markdown(chaine)

    st.subheader("Caractéristiques influençant le score")
    
#Affichage des graphes    
#graphes_streamlit(explanation)

    st.subheader("Définition des groupes")
    st.markdown("\
                \n\
    * Client : la valeur pour le client considéré\n\
    * Moyenne : valeur moyenne pour l'ensemble des clients\n\
    * En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
    * En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
    * Similaires : valeur moyenne pour les 20 clients les plus proches du client\
    considéré sur les critères sexe/âge/revenu/durée/montant du crédit\n\n\
        ")


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Paramètres du client')



#EXT_SOURCE_3=st.sidebar.slider('EXT_SOURCE_3', X.EXT_SOURCE_3.min(), X.EXT_SOURCE_3.max(), X.EXT_SOURCE_3.mean())
#EXT_SOURCE_2=st.sidebar.slider('EXT_SOURCE_2', X.EXT_SOURCE_2.min(), X.EXT_SOURCE_2.max(), X.EXT_SOURCE_2.mean())
#AMT_GOODS_PRICE=st.sidebar.slider('AMT_GOODS_PRICE', X.AMT_GOODS_PRICE.min(), X.AMT_GOODS_PRICE.max(), X.AMT_GOODS_PRICE.mean())
#AMT_CREDIT=st.sidebar.slider('AMT_CREDIT', X.AMT_CREDIT.min(), X.AMT_CREDIT.max(), X.AMT_CREDIT.mean())
# data = {'EXT_SOURCE_3': EXT_SOURCE_3,
#         'EXT_SOURCE_2': EXT_SOURCE_2,
#         'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
#         'AMT_CREDIT': AMT_CREDIT}
# client_parametres = pd.DataFrame(data, index=[0])


# def user_input():
#     EXT_SOURCE_3=st.sidebar.slider('EXT_SOURCE_3', X.EXT_SOURCE_3.min(), X.EXT_SOURCE_3.max(), X.EXT_SOURCE_3.mean())
#     EXT_SOURCE_2=st.sidebar.slider('EXT_SOURCE_2', X.EXT_SOURCE_2.min(), X.EXT_SOURCE_2.max(), X.EXT_SOURCE_2.mean())
#     AMT_GOODS_PRICE=st.sidebar.slider('AMT_GOODS_PRICE', X.AMT_GOODS_PRICE.min(), X.AMT_GOODS_PRICE.max(), X.AMT_GOODS_PRICE.mean())
#     AMT_CREDIT=st.sidebar.slider('AMT_CREDIT', X.AMT_CREDIT.min(), X.AMT_CREDIT.max(), X.AMT_CREDIT.mean())
#     data = {'EXT_SOURCE_3': EXT_SOURCE_3,
#             'EXT_SOURCE_2': EXT_SOURCE_2,
#             'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
#             'AMT_CREDIT': AMT_CREDIT}
#     client_parametres = pd.DataFrame(data, index=[0])
#     return client_parametres 

# df = user_input()
# st.subheader('on veut prédire l octroi du crédit')
# st.write(df)
# # Main Panel
# # Print specified input parameters
# st.header('Veuillez indiquer les paramètres de votre client')
# st.write()
# st.write('---')
# preprocess
#loaded_model = joblib.load('preprocessor.pkl')
# # Reads in saved classification model
#loaded_model = joblib.load('model.pkl')

# # Apply model to make predictions


# # Explaining the model's predictions using SHAP values
