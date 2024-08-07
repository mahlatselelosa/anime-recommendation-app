import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pickle

# Load data
@st.cache_data
def load_anime_data(file):
    return pd.read_csv(file)

@st.cache_data
def load_train_data(file):
    return pd.read_csv(file)

# Content-Based Filtering
def content_based_recommendations(user_input, data):
    data['genre'] = data['genre'].fillna('')
    data['combined_features'] = data['genre'] + ' ' + data['name']
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)

    def get_title_from_index(index):
        return data[data.index == index]['name'].values[0]

    def get_index_from_title(title):
        return data[data['name'] == title].index.values[0]

    recommendations = []
    for movie in user_input:
        try:
            movie_index = get_index_from_title(movie)
            similar_movies = list(enumerate(cosine_sim[movie_index]))
            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
            for element in sorted_similar_movies[1:11]:
                recommendations.append(get_title_from_index(element[0]))
        except IndexError:
            recommendations.append("Movie not found in dataset")
    return recommendations

# Collaborative-Based Filtering
def collaborative_based_recommendations(user_input, data, model_path):
    # Load pickled model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Prepare data
    data['rating'] = data['rating'].fillna(0)
    data = data.astype({'anime_id': 'int64'})
    data = data.drop_duplicates(['anime_id', 'rating'])
    data_pivot = data.pivot_table(index='anime_id', columns='name', values='rating').fillna(0)

    # Prepare user input
    user_input_ids = [data_pivot.columns.get_loc(title) for title in user_input if title in data_pivot.columns]
    user_vector = np.zeros(len(data_pivot.columns))
    user_vector[user_input_ids] = 1

    # Predict using the collaborative model
    user_vector = user_vector.reshape(1, -1)
    predictions = model.predict(user_vector)

    # Get recommendations
    recommendation_scores = list(enumerate(predictions[0]))
    recommendation_scores = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)
    recommended_anime = [data_pivot.columns[i] for i, _ in recommendation_scores[:10]]

    return recommended_anime

# Main Streamlit App
def main():
    # Add background color and styling using st.markdown with HTML and CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #e0f7fa;
            background-image: url('https://example.com/anime-background.jpg');
            background-size: cover;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(#ffeb3b,#f57c00);
            color: white;
        }
        h1 {
            color: #f44336;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        h2 {
            color: #ff9800;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        h3 {
            color: #2196f3;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        .css-1avcm0n {
            color: #4caf50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a page:", ['Team Page', 'Project Overview', 'Anime Recommendation'])

    if options == 'Team Page':
        st.title("Meet the Team")
        st.markdown("""
        ### Team Name
        - **Mahlatse Lelosa**
        - **Karabo Mathibela**
        - **Keneilwe Rangwaga**
        - **Koena Macdonald**
        - **Muwanwa Tshikovhi** 
        
        ### Contact Information
        - Email: mahlatselelosa98@gmail.com
                 tshikovhimuwanwa@gmail.com
                 kmahladisa9@gmail.com
                 karabomathibela44@gmail.com
                 patricia001105@gmail.com            
        """)
    elif options == 'Project Overview':
        st.title("Project Overview")
        st.markdown("""
        ## Objective
        The goal of this project is to develop an anime recommendation system using both content-based and collaborative filtering techniques.

        ## Methodology
        - **Data Collection**: We collected data from Kaggle, including user ratings and anime metadata.
        - **Data Preprocessing**: The data was cleaned and processed to ensure accuracy and consistency.
        - **Recommendation Algorithms**: We implemented two types of recommendation algorithms:
          - **Content-Based Filtering**: Recommends anime similar to the ones a user likes based on genre and other features.
          - **Collaborative-Based Filtering**: Recommends anime based on the preferences of other users who have similar tastes.
        
        ## Future Work
        - Improve the collaborative filtering algorithm using a trained model.
        - Implement a feedback loop to refine recommendations based on user input.
        - Expand the dataset with more comprehensive anime information.
        """)

    elif options == 'Anime Recommendation':
        st.title('Anime Recommendation App')

        # Display an image
        image_path = 'anime.jpg'  # Updated to absolute path
        image = Image.open(image_path)
        st.image(image)

        # File upload for anime.csv
        anime_file = st.file_uploader("Upload your anime dataset CSV file (anime.csv)", type="csv")
        if anime_file is not None:
            anime_data = load_anime_data(anime_file)

            # Initialize recommendations
            recommendations = []

            # User input section
            st.header('Select an algorithm')
            algorithm = st.radio(
                '',
                ('Content Based Filtering', 'Collaborative Based Filtering')
            )

            st.header('Enter Your Three Favorite Anime')
            user_input = []
            user_input.append(st.text_input('Enter first anime choice'))
            user_input.append(st.text_input('Enter second anime choice'))
            user_input.append(st.text_input('Enter third anime choice'))

            # Recommend button
            if st.button('Recommend'):
                if algorithm == 'Content Based Filtering':
                    recommendations = content_based_recommendations(user_input, anime_data)
                else:
                    # File upload for train.csv
                    train_file = st.file_uploader("Upload your training data CSV file (train.csv)", type="csv")
                    if train_file is not None:
                        train_data = load_train_data(train_file)
                        # Path to pickled model
                        model_path = 'C:/Users/mahla/Downloads/model/collaborative_model.pkl'  # Use the correct absolute path
                        recommendations = collaborative_based_recommendations(user_input, train_data, model_path)
                    else:
                        st.error("Please upload the training data CSV file.")

                st.write('Here are your recommendations:')
                for i, rec in enumerate(recommendations):
                    st.write(f"{i + 1}. {rec}")

if __name__ == '__main__':
    main()
