# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:02:43 2021

@author: glass
"""

# voici le code permettant de générer le Dashboard via Streamlit

# mise en place de l'environnement Python
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
import plotly.graph_objects as go
from sklearn.ensemble import StackingClassifier


st.write("""
# "Prêt à dépenser" Prediction App
Cette application prédit l'octroi du crédit à un client en fonction de son profil!
""")
st.write('---')

# Loads the pret a depenser Dataset
PATH = os.getcwd()
PATH += "\\"
f = PATH+'application_train.csv'

df = pd.read_csv(f,
                 low_memory=False,
                 verbose=False,
                 encoding='ISO-8859-1',
                 dtype={'Special': 'object'}
                 )

# @st.cache #mise en cache de la fonction pour exécution unique
# def chargement_explanation(id_input, df, model, sample):
#     return interpretation(str(id_input), df, model, sample=sample)


# @st.cache #mise en cache de la fonction pour exécution unique
# def chargement_ligne_data(id, df):
#     return df[df['SK_ID_CURR']==int(id)].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

liste_id = df['SK_ID_CURR'].tolist()

df_pay = df[df.TARGET == 0]
df_unpay = df[df.TARGET == 1]
X = df.drop(columns=['TARGET'])
y = df['TARGET']

print(X.shape)



#affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
chaine = "l'id Saisi est " + str(id_input)
st.write(chaine)

sample_en_regle = str(list(df[df['TARGET'] == 0].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples d\'id de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(df[df['TARGET'] == 1].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_defaut)
    st.write(chaine_en_regle)

elif (int(id_input) in liste_id): 

#affichage de la jauge de probabilité

    

# affichage des informations clients

# st.header("Informations client")

# colonnes = ["SK_ID_CURR","CODE_GENDER","AGE","CNT_CHILDREN","AMT_INCOME_TOTAL","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","DAYS_EMPLOYED",
#             "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","ORGANIZATION_TYPE",'AMT_CREDIT_ACTIVE']

    # #Appel de l'API : indiquer l'url de l'API 

    # API_url = "http://127.0.0.1:5000/credit/" + id_input

    # with st.spinner('Chargement du score du client...'):
    #     json_url = urlopen(API_url)

    #     API_data = json.loads(json_url.read())
    #     classe_predite = API_data['prediction']
    #     if classe_predite == 1:
    #         etat = 'client à risque'
    #     else:
    #         etat = 'client peu risqué'
    #     proba = 1-API_data['proba'] 

    #     #affichage de la prédiction
    #     prediction = API_data['proba']
    #     classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
    #     classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
    #     chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'

    #st.markdown(chaine)

    st.subheader("Caractéristiques influençant le score")


    #chargement du preprocessor
    loaded_preprocessor = joblib.load('preprocessor.pkl')
    #chargement du modèle
    loaded_model = joblib.load('model.pkl')
    
    data_clientunique = X[X['SK_ID_CURR']==int(id_input)]
    
    st.write(data_clientunique.shape)
    
    
    data_clientunique=loaded_preprocessor.transform(data_clientunique)
    st.write(data_clientunique.shape)
    score_client=loaded_model.predict_proba(data_clientunique)
    st.write(score_client[0])
    
    jauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = score_client[0][0]*100,
    mode = "gauge+number",
    title = {'text': "Probabilité de défaut de paiement (en pourcentage)"},
    gauge = {'axis': {'range': [None, 100]},
              'bar': {'color': "darkgrey"},
              'bgcolor': "red",
              'steps' : [
                  {'range': [0, 35], 'color': "green"},
                  {'range': [35, 60], 'color': "orange"}],
              'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))

    st.write(jauge)
    
#affichage de l'explication du score
# with st.spinner('Chargement des détails de la prédiction...'):
#     explanation = chargement_explanation(str(id_input), df,loaded_model,sample=False)
#     st.success('Done!')
    
#Affichage des graphes    
# graphes_streamlit(explanation)

# st.subheader("Définition des groupes")
# st.markdown("\
    #             \n\
    # * Client : la valeur pour le client considéré\n\
    # * Moyenne : valeur moyenne pour l'ensemble des clients\n\
    # * En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
    # * En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
    # * Similaires : valeur moyenne pour les 20 clients les plus proches du client\
    # considéré sur les critères sexe/âge/revenu/durée/montant du crédit\n\n\
    #     ")

    #Affichage du dataframe d'explicabilité
    #st.write(explanation)

    #Détail des explications
st.subheader('Traduction des explication')
# chaine_explanation, df_explanation = df_explain(explanation)
# chaine_features = '\n\
#     '
# for x, y in zip(df_explanation['Feature'], df_explanation['Nom francais']):
#     chaine_features += '* **' + str(x) + ' ** '+str(y) +'\n'\''
# st.markdown(chaine_features)

# st.write(df_explanation, unsafe_allow_html=True)

    #Modifier le profil client en modifiant une valeur
st.subheader('Modifier le profil client')
st.sidebar.header("Modifier le profil client")
st.sidebar.markdown('Cette section permet de modifier une des valeurs les plus caractéristiques du client et de recalculer son score')
# features = explanation['feature'].values.tolist()
# liste_features = tuple([''] + features)
# feature_to_update = ''
# feature_to_update = st.sidebar.selectbox('Quelle caractéristique souhaitez vous modifier', liste_features)

    # #st.write(dataframe.head())

    # if feature_to_update != '':
    #     value_min = dataframe[feature_to_update].min()
    #     value_max = dataframe[feature_to_update].max()
    #     #st.write(list(explanation['feature'].values))
    #     #st.write(explanation['feature'].values[0])
    #     default_value = explanation[explanation['feature'] == feature_to_update]['customer_values'].values[0]
    #     #st.write(default_value)


    #     min_value = float(dataframe[feature_to_update].min())
    #     max_value = float(dataframe[feature_to_update].max())

    #     if (min_value, max_value) == (0,1): 
    #         step = float(1)
    #     else :
    #         step = float((max_value - min_value) / 20)

    #     update_val = st.sidebar.slider(label = 'Nouvelle valeur (valeur d\'origine : ' + str(default_value)[:4] + ')',
    #         min_value = min_value,
    #         max_value =max_value,
    #         value = default_value,
    #         step = step)

    #     if update_val != default_value:
    #         time.sleep(0.5)
    #         update_predict, proba_update = predict_update(id_input, dataframe, feature_to_update, update_val)
    #         if update_predict == 1:
    #             etat = 'client à risque'
    #         else:
    #             etat = 'client peu risqué'
    #         chaine = 'Nouvelle prédiction : **' + etat +  '** avec **' + str(round((proba_update[0][1])*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'
    #         st.sidebar.markdown(chaine)    


    # st.subheader('Informations relatives au client')
    # df_client = chargement_ligne_data(id_input, df).T
    # df_client['nom_fr'] = [correspondance_feature(feature) for feature in df_client.index]
    # st.write(df_client)
        

    # #else: 
    # st.write('Identifiant non reconnu')



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