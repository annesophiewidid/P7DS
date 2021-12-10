# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:02:43 2021

@author: glass
"""

# voici le code permettant de générer le Dashboard via Streamlit

# mise en place de l'environnement Python
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn import metrics
import numpy as np
import plotly.graph_objects as go
import urllib
import pickle
import seaborn as sns

st.set_page_config(page_title='Loan application scoring dashboard',
                       page_icon='random',
                       layout='centered',
                       initial_sidebar_state='auto')



st.write("""
# "Prêt à dépenser" Prediction App
Cette application prédit la probabilité de défaut de paiement si vous octroyez un crédit à un client en fonction de son profil!
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

liste_id = df['SK_ID_CURR'].tolist()

df_pay = df[df.TARGET == 0]
df_unpay = df[df.TARGET == 1]
X = df.drop(columns=['TARGET'])
y = df['TARGET']

#affichage formulaire
st.title('Calculez le score de votre client')
st.subheader("Prédictions de scoring client")
id_input = st.text_input('Veuillez saisir l\'identifiant de\'votre client:', )
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

    st.subheader("Probabilité que votre client soit en défaut de paiement (classe 1) ou non (classe 0)")

#chargement du preprocessor
    loaded_preprocessor = joblib.load('preprocessor.pkl')
    
#chargement du modèle
    loaded_model = joblib.load('model.pkl')
    
    data_clientunique = X[X['SK_ID_CURR']==int(id_input)]
    
    #st.write(data_clientunique.shape)     
    data_clientunique=loaded_preprocessor.transform(data_clientunique)
    
    #st.write(data_clientunique.shape)
    score_client=loaded_model.predict_proba(data_clientunique)
    st.write(score_client[0])
    
#affichage de la jauge de probabilité  
    jauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = score_client[0][1]*100,
    mode = "gauge+number",
    title = {'text': "Jauge de Probabilité de défaut de paiement (en pourcentage)"},
    gauge = {'axis': {'range': [None, 100]},
              'bar': {'color': "darkgrey"},
              'bgcolor': "red",
              'steps' : [
                  {'range': [0, 35], 'color': "green"},
                  {'range': [35, 60], 'color': "orange"}],
              'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
    st.write(jauge)
    
    
    st.subheader('Traduction des explications avec Shap')

#chargement du modèle
    explainer = shap.TreeExplainer(loaded_model)
#calcul des valeurs Shap
    shap_values = explainer.shap_values(data_clientunique)
    
# Chargement de la liste de features
    PATH = os.getcwd()
    PATH += "\\"
    h = PATH+'FEATURELIST.csv'

    FEATURELIST = pd.read_csv(h,
                  low_memory=False,
                  verbose=False,
                  encoding='ISO-8859-1',
                  dtype={'Special': 'object'})
    
    FEATURENAMES = FEATURELIST["0"].tolist()

#Affichage des graphes
    shap.initjs()
    fig=shap.force_plot(explainer.expected_value[1],shap_values[1][0],feature_names=FEATURENAMES, show=False, matplotlib=True)   
    st.pyplot(fig)
    
    nbr = df_pay[['SK_ID_CURR','AMT_CREDIT']].groupby('AMT_CREDIT').count().sort_values(by='SK_ID_CURR', ascending=False)
    nbr.reset_index(0, inplace=True)
    nbr.rename(columns={'SK_ID_CURR':'nombre'}, inplace=True)
    
    fig1, axes = plt.subplots(nrows=1,ncols=2, sharex=False, sharey=False, figsize=(20,8))
    fig1=sns.histplot(data=nbr, x="AMT_CREDIT", kde=True, ax=axes[0], color="#00afe6", alpha=0.6)
    axes[0].set_title("AMT_CREDIT", color='#2cb7b0')
    fig1, ax = plt.subplots()    
    
    st.pyplot(fig1)




