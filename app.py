import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a
model = joblib.load(open("model-v2.joblib", "rb"))
dfad = pd.read_csv(r'address.csv')


def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.furnishing = df.furnishing.map(
        {'unfurnished': 0, 'semifurnished': 1, 'furnished': 2})
    dfad2 = pd.Series(dfad.addval.values, index=dfad.address).to_dict()
    # st.write(dfad2)
    df.address = df.address.map(dfad2)

    return df


def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(
        data=data, columns=['Percentage'], index=['Low', 'Ave', 'High'])
    ax = grad_percentage.plot(kind='barh', figsize=(
        7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")

    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed',
                   alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level",
                  labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Wine Quality", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None,
                 loc='center', pad=None, weight='bold')

    st.pyplot()
    return


st.write("""
# Rent Prediction ML Web-App 
This app predicts the ** Approximate rent **  using **housing features** input via the **side panel** 
""")


# user input parameter collection with streamlit side bar
st.sidebar.header('User Input Parameters')


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    # df = pd.read_csv(r'address.csv')
    # print(df)
    address = st.sidebar.selectbox(
        "Select location count", dfad['address'])
    furnishing = st.sidebar.selectbox(
        "Select furnishing status", ('unfurnished', 'semifurnished', 'furnished'))

    bedroom = st.sidebar.slider('Select bathroom count', 1, 6, 2)
    bathrooms = st.sidebar.slider('Select bathroom count', 1, 7, 2)
    area = st.sidebar.slider('Select area', 500, 10000, 1000)
    floor_number = st.sidebar.slider('Select floor number', 0, 8, 0)
    parking = st.sidebar.slider('Select parking', 0, 6, 1)
    wheelchairadption = st.sidebar.slider('Select wheelchair adption', 0, 1, 0)
    petfacility = st.sidebar.slider('Select pet facility', 0, 1, 0)
    powerbackup = st.sidebar.slider('Select power backup', 0, 2, 0)
    servant_room = st.sidebar.slider('Select servant room', 0, 1, 0)
    no_room = st.sidebar.slider('Select aditional room', 0, 1, 0)
    deposit_amt = st.sidebar.slider('Select deposit', 0, 1500000, 1000)
    mnt_amt = st.sidebar.slider('Select maintenance amt', 0, 40000, 1000)
    brok_amt = st.sidebar.slider('Select broker amt', 0, 275000, 1000)

    features = {'furnishing': furnishing,
                'bedroom': bedroom,
                'area': area,
                'bathrooms': bathrooms,
                'floor_number': floor_number,
                'parking': parking,
                'wheelchairadption': wheelchairadption,
                'petfacility': petfacility,
                'powerbackup': powerbackup,
                'servant_room': servant_room,
                'no_room': no_room,
                'deposit_amt': deposit_amt,
                'mnt_amt': mnt_amt,
                'brok_amt': brok_amt,
                'address': address
                }
    data = pd.DataFrame(features, index=[0])

    return data


user_input_df = get_user_input()
# user_input_df = dict(bedroom=1, bathrooms=1, area=500, furnishing=0, floor_number=2, parking=1, wheelchairadption=1, petfacility=1,
#  powerbackup=1, no_room=1, servant_room=1, maintenance_amt=500, brok_amt=1000, deposit_amt=20000, mnt_amt=1000,)
processed_user_input = data_preprocessor(user_input_df)
df2 = pd.DataFrame(user_input_df, index=[0])
st.subheader('User Input parameters')
st.write(df2)

prediction = model.predict(df2)
st.write(prediction)
# prediction_proba = model.predict_proba(df2)

# visualize_confidence_level(prediction_proba)
