import streamlit as st

st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings

warnings.filterwarnings('ignore')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(
        'anime-dataset-2023.csv',
        low_memory=False,
        usecols=['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Aired', 'Producers', 'Licensors',
                 'Studios', 'Source', 'Synopsis', 'Rating', 'Popularity', 'Members']
    )

anime_data = load_data()

# Extract year from 'Aired'
def extract_year(date_str):
    if 'Unknown' in date_str:
        return np.nan
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', date_str)
    if len(years) == 2:
        return (int(years[0]) + int(years[1])) // 2
    elif years:
        return int(years[0])
    else:
        return np.nan

anime_data['Aired'] = anime_data['Aired'].apply(extract_year)
middle_year = anime_data['Aired'].median()
anime_data['Aired'].fillna(middle_year, inplace=True)
anime_data['Aired'] = anime_data['Aired'].astype(int)

# Clean text
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', "I'", text)
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

anime_data['Name'] = anime_data['Name'].apply(text_cleaning)

# Process synopsis
tfidf = TfidfVectorizer(stop_words='english')
anime_data['Synopsis'] = anime_data['Synopsis'].fillna('').str.lower()
anime_data['Synopsis'] = anime_data['Synopsis'].str.replace(r'[^\w\s]+', '', regex=True)

# Create TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(anime_data['Synopsis'])
cosine_synopsis = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(anime_data.index, index=anime_data['Name']).drop_duplicates()

# Recommendation functions
def get_recommendations(title, cosine_sim=cosine_synopsis, suggest_amount=6):
    try:
        idx = indices[title]
    except KeyError:
        st.error(f"Anime '{title}' not found.")
        return None
    return get_recommendations_by_id(idx, cosine_sim, suggest_amount)

def get_recommendations_by_id(idx, cosine_sim=cosine_synopsis, suggest_amount=6):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:suggest_amount]
    anime_indices = [i[0] for i in sim_scores]
    anime_names = anime_data.iloc[anime_indices]['Name'].tolist()
    return anime_names


# Custom CSS
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f7f7f7;
        }
        .main {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# Header
st.title("Anime Recommendation System")
st.markdown(
    "<h2 style='color: #4CAF50;'>Find your next favorite anime based on what you love!</h2>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "Welcome to the Anime Recommendation System,your go-to tool for discovering anime tailored to your preferences.Simply enter an anime you love, and get personalized suggestions to explore next.Whether you're a longtime fan or new to anime, there's something here for everyone!"
)

# User input
title_input = st.text_input(
    "Enter the name of an anime:"
)
suggest_amount = st.slider("Number of recommendations:", min_value=1, max_value=10, value=6)

if st.button("Get Recommendations"):
    if title_input:
        recommendations = get_recommendations(title_input, suggest_amount=suggest_amount + 1)
        if recommendations is not None:
            st.markdown(f"<h3 style='color: #4CAF50;'>Recommendations for {title_input}:</h3>", unsafe_allow_html=True)
            for i, anime in enumerate(recommendations, start=1):
                st.markdown(f"{i}. {anime}")
    else:
        st.error("Please enter a valid anime name.")
