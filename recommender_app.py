import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load data and model (ensure correct file paths)
data = pickle.load(open('data.pkl', 'rb'))
RF = pickle.load(open('random_forest_model.pkl', 'rb'), encoding='latin1')

songs = data['name'].unique().tolist()  # Convert to list for st.selectbox compatibility

def recommend(user_song_name):
    if user_song_name in data['name'].values:  # Check if song exists
        user_song_index = data[data['name'] == user_song_name].index[0]
        user_song_features = data.iloc[user_song_index, :].drop('name')  # Exclude 'name' for prediction

        # Predict popularity for chosen song and other songs
        user_song_pred = RF.predict(user_song_features.values.reshape(1, -1))[0]
        song_preds = RF.predict(data.drop('name', axis=1))  # Exclude 'name' for prediction

        # Recommend top N songs with similar predicted popularity
        N = 10
        diffs = np.abs(song_preds - user_song_pred)
        closest_indices = diffs.argsort()[:N]
        recommended_songs = data.loc[data.index[closest_indices], 'name'].tolist()
        return recommended_songs
    else:
        return "Sorry, that song is not in our database."

st.title('Song Recommendation Engine')

selected_song = st.selectbox('Choose a song you like:', songs)

if st.button('Recommend'):
    recommendations = recommend(selected_song)
    if isinstance(recommendations, list):  # Check if recommendations are available
        st.header("Recommendations for you:")
        for song in recommendations:
            st.write(song)
    else:
        st.write(recommendations)  # Display error message if song not found
