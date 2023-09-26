import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from PIL import Image
from io import BytesIO
import base64
import requests
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import math
global output

def open_spot():
    # Replace with your own Client ID and Client Secret
    CLIENT_ID = "2ce05a82e04546f1a45d33e19f8d2f45"
    CLIENT_SECRET = "5e1cfa53241449de99e9cf89329cdd69"

    # Base64 encode the client ID and client secret
    client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    client_credentials_base64 = base64.b64encode(client_credentials.encode())

    # Request the access token
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f'Basic {client_credentials_base64.decode()}'
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(token_url, data=data, headers=headers)

    if response.status_code == 200:
        access_token = response.json()['access_token']
        print("Access token obtained successfully.")
    else:
        print("Error obtaining access token.")
        exit()

    return access_token

access_token = open_spot()
all_tracks = pd.read_csv("data/data.csv")
all_tracks.drop("year", axis = 1, inplace = True)

all_tracks = all_tracks.sort_values(by='popularity', ascending=False)
scaler = MinMaxScaler()

def get_trending_playlist_data(playlist_id, access_token):
    # Set up Spotipy with the access token
    sp = spotipy.Spotify(auth=access_token)

    # Extract relevant information and store in a list of dictionaries
    music_data = []
    i = 0
    while len(sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))', offset = i * 100)['items']):
        playlist_tracks = sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))', offset = i * 100)
        for track_info in tqdm(playlist_tracks['items']):
            try:
                track = track_info['track']
                track_name = track['name']
            except:
                continue

            artists = [artist['name'] for artist in track['artists']]
            album_id = track['album']['id']
            track_id = track['id']

            if track_name in all_tracks['name'].values:
                music_data.append(all_tracks.loc[all_tracks['name'] == track_name].iloc[0].squeeze().to_dict())
                continue

            # Get audio features for the track
            audio_features = sp.audio_features(track_id)[0] if track_id != 'Not available' else None

            # Get release date of the album
            try:
                album_info = sp.album(album_id) if album_id != 'Not available' else None
                release_date = album_info['release_date'] if album_info else None
            except:
                release_date = None

            # Get popularity of the track
            try:
                track_info = sp.track(track_id) if track_id != 'Not available' else None
                popularity = track_info['popularity'] if track_info else None
            except:
                popularity = None

            # Add additional track information to the track data
            track_data = {
                'name': track_name,
                'artists': artists,
                'popularity': popularity,
                'release_date': release_date,
                'duration_ms': audio_features['duration_ms'] if audio_features else None,
                'explicit': track_info.get('explicit', None),
                'danceability': audio_features['danceability'] if audio_features else None,
                'energy': audio_features['energy'] if audio_features else None,
                'key': audio_features['key'] if audio_features else None,
                'loudness': audio_features['loudness'] if audio_features else None,
                'mode': audio_features['mode'] if audio_features else None,
                'speechiness': audio_features['speechiness'] if audio_features else None,
                'acousticness': audio_features['acousticness'] if audio_features else None,
                'instrumentalness': audio_features['instrumentalness'] if audio_features else None,
                'liveness': audio_features['liveness'] if audio_features else None,
                'valence': audio_features['valence'] if audio_features else None,
                'tempo': audio_features['tempo'] if audio_features else None,
                # Add more attributes as needed
            }

            music_data.append(track_data)

        i += 1

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(music_data)

    return df

def similarity(music_df, music_df_2):
    music_features = music_df[['danceability', 'energy', 
            'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']].values
    music_features_scaled = scaler.fit_transform(music_features)

    music_features_2 = music_df_2[['danceability', 'energy', 
            'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']].values
    music_features_scaled_2 = scaler.fit_transform(music_features_2)

    music_mean_1 = music_features_scaled.mean(axis = 0)
    music_mean_2 = music_features_scaled_2.mean(axis = 0)

    cos_sim = np.diag(cosine_similarity(music_features_scaled, music_features_scaled_2))
    cos_sim = np.where(cos_sim > 1, 1, cos_sim)

    return np.mean(90 - (np.arccos(cos_sim)) * 57.2958) * 100/90

def content_based_recommendations(music_df, num_recommendations=5):
    music_features = music_df[['danceability', 'energy', 
            'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']].values
    music_features_scaled = scaler.fit_transform(music_features)

    all_tracks_features = all_tracks[['danceability', 'energy', 
            'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']].values
    all_tracks_features_scaled = scaler.fit_transform(all_tracks_features)

    # Calculate the similarity scores based on music features (cosine similarity)
    similarity_scores = cosine_similarity(music_features_scaled, all_tracks_features_scaled)

    # Get the indices of the most similar songs
    similar_song_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations * 5 + 1]

    artists = [item for sublist in music_df["artists"] for item in sublist]

    # Get the names of the most similar songs based on content-based filtering
    content_based_rec = all_tracks.iloc[similar_song_indices][['name', 'artists', 'release_date', 'popularity']]
    #content_based_rec.loc[len(set(content_based_rec["artists"]).intersection(set(artists))) == 0, 'popularity'] /= 1.2
    content_based_rec = content_based_rec.sort_values(by="popularity", ascending = False)
    content_based_rec = content_based_rec.loc[~content_based_rec['name'].isin(music_df['name'])].copy()

    return content_based_rec.head(num_recommendations)

def id_from_url(url):
    str1 = url.split("playlist/")[1]
    return str1.split("?")[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Spotify.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        playlist_id_1 = id_from_url(request.form["play1"])
        playlist_id_2 = id_from_url(request.form["play2"])

        all_tracks = pd.read_csv("data/data.csv")
        all_tracks.drop("year", axis = 1, inplace = True)

        all_tracks = all_tracks.sort_values(by='popularity', ascending=False)

        music_df = get_trending_playlist_data(playlist_id_1, access_token)
        music_df_2 = get_trending_playlist_data(playlist_id_2, access_token)
        all_tracks = all_tracks.head(10000)
        
        output = similarity(music_df, music_df_2)

        music_combined = pd.concat([music_df, music_df_2])
        def clean_list(str_list):
            for char in "[]'":
                str_list = str_list.replace(char, '')
            return str_list

        df = content_based_recommendations(music_combined, num_recommendations=10)
        df["artists"] = df["artists"].apply(clean_list)

        return render_template('Spotify.html', tables = [df.to_html(index = False, classes = 'data', header = 'true')], prediction_text=f"Similarity = {round(output, 2)}%")
    
    except:
        return render_template('Spotify.html')

@app.route('/results',methods=['POST'])
def results():
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
