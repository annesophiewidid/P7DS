# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:02:43 2021

@author: glass
"""

# voici le code permettant de générer le Dashboard via Streamlit

# mise en place de l'environnement Python
import os
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from plotly import graph_objs as go
import requests
from subprocess import Popen
import time


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
# PATH = os.getcwd()
# f = os.path.join(PATH,"application_train.csv")


df = pd.read_csv("application_train.csv",
                  low_memory=False,
                  verbose=False,
                  encoding='UTF-8',
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
id_input = st.text_input('Veuillez saisir l\'identifiant de\ votre client:', )
chaine = "l'id Saisi est " + str(id_input)

st.write(chaine)

sample_en_regle = str(list(df[df['TARGET'] == 0].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples d\'id de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(df[df['TARGET'] == 1].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_defaut)
    st.write(chaine_en_regle)
    
elif (id_input not in liste_id):   
    st.write('je ne connais pas cet id')

elif (id_input in liste_id): 

    st.subheader("Probabilité que votre client soit en défaut de paiement (classe 1) ou non (classe 0)")

#chargement du preprocessor
    loaded_preprocessor = joblib.load('preprocessor.pkl')
    

#FONCTIONNEMENT SANS API FLASK 
#chargement du modèle
    # loaded_model = joblib.load('model.pkl')
    
    # data_clientunique = X[X['SK_ID_CURR']==int(id_input)]
     
    # data_clientunique=loaded_preprocessor.transform(data_clientunique)
        
    # score_client=loaded_model.predict_proba(data_clientunique)
    
    # jauge = go.Figure(go.Indicator(
    # domain = {'x': [0, 1], 'y': [0, 1]},
    # value = score_client[0][1]*100,
    # mode = "gauge+number",
    # title = {'text': "Jauge de Probabilité de défaut de paiement (en pourcentage)"},
    # gauge = {'axis': {'range': [None, 100]},
    #           'bar': {'color': "darkgrey"},
    #           'bgcolor': "red",
    #           'steps' : [
    #               {'range': [0, 35], 'color': "green"},
    #               {'range': [35, 60], 'color': "orange"}],
    #           'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
    # st.write(jauge)
   
#FONCTIONNEMENT avec API FLASK 
    
    # url=f"http://127.0.0.1:5000/prediction/{id_input}/"
    
    url= f"http://annesophiegimenez.pythonanywhere.com/prediction/{id_input}/"
    
    # try:
    #     response=requests.get(url)
    # except requests.exceptions.ConnectionError:
    #     st.error("flask n est pas allumé")
    #     process = Popen(["python", "api.py"], shell=True)
    #     time.sleep(5)
    
    response=requests.get(url)
        
    score_client=float(response.content)
    

       
    
    jauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = score_client,
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
    loaded_preprocessor = joblib.load('preprocessor.pkl')    
    loaded_model = joblib.load('model.pkl')
    data_clientunique = X[X['SK_ID_CURR']==(id_input)]
    data_clientunique=loaded_preprocessor.transform(data_clientunique)    
    
#calcul des valeurs Shap
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(data_clientunique)
    
# Chargement de la liste de features
    # PATH = os.getcwd()
    # PATH += "\\"
    # h = PATH+'FEATURELIST.csv'

    FEATURELIST = pd.read_csv("FEATURELIST.csv",
                  low_memory=False,
                  verbose=False,
                  encoding='ISO-8859-1',
                  dtype={'Special': 'object'})
    
    FEATURENAMES = FEATURELIST["0"].tolist()

#Affichage des graphes

#visualisation des variables les + importantes pour ce client
    st.title('quelles variables expliquent ce score ?')
    shap.initjs()
    fig=shap.force_plot(explainer.expected_value[1],shap_values[1][0],feature_names=FEATURENAMES, show=False, matplotlib=True)   
    st.pyplot(fig)
    
#SELECTION D UNE VARIABLE POUR COMPARER CE CLIENT AUX AUTRES

#AMT_CREDIT
if st.button('Montant du crédit - AMT CREDIT'):

    # Data   
    credit = df[['AMT_CREDIT','SK_ID_CURR']].sort_values(by=['AMT_CREDIT'])
    
    # Plot
    fig, ax = plt.subplots(figsize = (12, 12))
    plt.title('répartition clients par montant du crédit')
    ax.hist(credit['AMT_CREDIT'], bins=100,edgecolor="black")
    
    # Label and coordinate
    z=credit.loc[df['SK_ID_CURR'] == (id_input)]
    a=z['AMT_CREDIT'].iloc[0]
    plt.axvline(a, color='r')
     
    # Show the graph
    st.pyplot(fig)
    
#DAYS_BIRTH
if st.button('Age du client - DAYS_BIRTH'):
    
    # Data   
    daysbirth = df[['DAYS_BIRTH','SK_ID_CURR']].sort_values(by=['DAYS_BIRTH'])    
        
    # Plot
    fig, ax = plt.subplots(figsize = (12, 12))
    plt.title('répartition clients par age')
    ax.hist(daysbirth['DAYS_BIRTH'], bins=100,edgecolor="black")
    
    # Label and coordinate
    z=daysbirth.loc[df['SK_ID_CURR'] == (id_input)]
    a=z['DAYS_BIRTH'].iloc[0]
    plt.axvline(a, color='r')
     
    # Show the graph
    st.pyplot(fig)
    
   #EXT_SOURCE_3
if st.button('Source externe 3 - EXT_SOURCE_3'):
     
    # Data   
     extsource3 = df[['EXT_SOURCE_3','SK_ID_CURR']].sort_values(by=['EXT_SOURCE_3'])     
     # Plot
     fig, ax = plt.subplots(figsize = (12, 12))
     plt.title('répartition clients par EXT_SOURCE_3')
     ax.hist(extsource3['EXT_SOURCE_3'], bins=100,edgecolor="black")
    
     # # Label and coordinate
     z=extsource3.loc[df['SK_ID_CURR'] == (id_input)]
     a=z['EXT_SOURCE_3'].iloc[0]
     plt.axvline(a, color='r')
     
     # Show the graph
     st.pyplot(fig)
     
        #DAYS_ID_PUBLISH 
if st.button('date de MAJ de la pièce identité client - DAYS_ID_PUBLISH'):
     # Data   
     DAYS_IDPUBLISH = df[['DAYS_ID_PUBLISH','SK_ID_CURR']].sort_values(by=['DAYS_ID_PUBLISH'])     
     # Plot
     fig, ax = plt.subplots(figsize = (12, 12))
     plt.title('répartition clients par DAYS_ID_PUBLISH')
     ax.hist(DAYS_IDPUBLISH['DAYS_ID_PUBLISH'], bins=100,edgecolor="black")
    
     # # Label and coordinate
     z=DAYS_IDPUBLISH.loc[df['SK_ID_CURR'] == (id_input)]
     a=z['DAYS_ID_PUBLISH'].iloc[0]
     plt.axvline(a, color='r')
     
     # Show the graph
     st.pyplot(fig)
     
#AMT_ANNUITY
if st.button('Annuité - AMT_ANNUITY'):

    # Data   
    AMTANNUITY = df[['AMT_ANNUITY','SK_ID_CURR']].sort_values(by=['AMT_ANNUITY'])
    
    # Plot
    fig, ax = plt.subplots(figsize = (12, 12))
    plt.title('répartition clients par annuité')
    ax.hist(AMTANNUITY['AMT_ANNUITY'], bins=100,edgecolor="black")
    
    # Label and coordinate
    z=AMTANNUITY.loc[df['SK_ID_CURR'] == (id_input)]
    a=z['AMT_ANNUITY'].iloc[0]
    plt.axvline(a, color='r')
     
    # Show the graph
    st.pyplot(fig)
    
