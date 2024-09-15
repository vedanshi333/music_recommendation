import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

# Load Data
tracks = pd.read_csv('data.csv')

# Data Preprocessing
label_encoder_name = LabelEncoder()
label_encoder_artists = LabelEncoder()

# Label encode 'artists' and 'name'
tracks['artists_encoded'] = label_encoder_artists.fit_transform(tracks['artists'])
tracks['name_encoded'] = label_encoder_name.fit_transform(tracks['name'])

# Define similarity function
def get_similarities(song_name, data):
    features = ['artists_encoded', 'instrumentalness', 'loudness', 'energy', 'year']
    array1 = data[data['name'] == song_name][features].to_numpy()
    similar = []
    
    for _, row in data.iterrows():
        array2 = row[features].to_numpy().reshape(1, -1)
        num_sim = cosine_similarity(array1, array2)[0][0]
        similar.append(num_sim)
    
    return similar

# Artist formatting function
def format_artists(artists):
    if isinstance(artists, str):
        try:
            artists_list = literal_eval(artists)
            if isinstance(artists_list, list):
                return ' & '.join(artists_list)
            else:
                return artists
        except (ValueError, SyntaxError):
            return artists
    elif isinstance(artists, list):
        return ' & '.join(artists)
    return artists

# Recommendation function
def recommend_songs(song_name, data=tracks):
    if tracks[tracks['name'] == song_name].shape[0] == 0:
        st.write('This song is either not so popular or you entered an invalid name.')
        st.write("Some songs you may like:")
        random_songs = data.sample(n=5)['name'].values
        for song in random_songs:
            st.write(song)
        return
    
    data['similarity_factor'] = get_similarities(song_name, data)
    data.sort_values(by=['similarity_factor', 'popularity'], ascending=[False, False], inplace=True)
    
    top_recommendations = data[['name', 'artists']][2:7]
    
    for _, row in top_recommendations.iterrows():
        formatted_artists = format_artists(row['artists'])
        st.markdown(f"<p style='color:green'>{row['name'].title()} by {formatted_artists}</p>", unsafe_allow_html=True)

# Streamlit App
st.markdown("<h1 style='color:green'>Song Recommendation System</h1>", unsafe_allow_html=True)

# Add an input box with suggestions (autocomplete)
song_names = tracks['name'].unique().tolist()

# Select or type a song name
song_input = st.selectbox("Enter a song name:", song_names, index=0)

# Add a button to trigger recommendations
if st.button("Get Recommendations"):
    recommend_songs(song_input)
