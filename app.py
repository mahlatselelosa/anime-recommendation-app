import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64
import pickle

# Function to add background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
        }}
        .sidebar .sidebar-content {{
            background: linear-gradient(#ffeb3b,#f57c00);
            color: white;
        }}
        h1 {{
            color: #ffff00;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }}
        h2 {{
            color: #ffff00;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }}
        h3 {{
            color: #ffff00;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }}
        .css-1avcm0n {{
            color: black;
        }}
        .black-text {{
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load data
@st.cache_data
def load_data(file):
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

# Collaborative-Based Filtering with Pickled Model (Placeholder)
def collaborative_based_recommendations(user_input, model_path, data):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    user_data = np.zeros((1, data.shape[1]))
    for title in user_input:
        try:
            idx = data.columns.get_loc(title)
            user_data[0, idx] = 1
        except KeyError:
            pass

    recommendations = model.predict(user_data)
    recommended_anime = [data.columns[i] for i in recommendations[0].argsort()[-10:][::-1]]
    
    return recommended_anime

# Main Streamlit App
def main():
    # Add background image
    add_bg_from_local('C:/Users/mahla/OneDrive/Desktop/Streamlit app/anime 5.jpg')

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a page:", ['Team Page', 'Project Overview', 'Anime Recommendation'])

    if options == 'Team Page':
        st.title("Meet the Team")
        st.markdown("""
        ### Team Name
        - <span class="black-text">**Mahlatse Lelosa**</span>
        - <span class="black-text">**Karabo Mathibela**</span>
        - <span class="black-text">**Keneilwe Rangwaga**</span>
        - <span class="black-text">**Josephine Ndukwani**</span>
        - <span class="black-text">**Koena Macdonald**</span>
        - <span class="black-text">**Mowanwa Tshikovhi**</span>
        
        ### Contact Information
        - <span class="black-text">Email: mahlatselelosa98@gmail.com</span>
        - <span class="black-text">Social Media: [LinkedIn](https://www.linkedin.com/in/mahlatse-lelosa-248476318/)</span>
        """, unsafe_allow_html=True)
    elif options == 'Project Overview':
        st.title("Project Overview")
        st.markdown("""
        ## Objective
        <span class="black-text">The goal of this project is to develop an anime recommendation system using both content-based and collaborative filtering techniques.</span>

        ## Methodology
        - <span class="black-text">**Data Collection**: We collected data from Kaggle, including user ratings and anime metadata.</span>
        - <span class="black-text">**Data Preprocessing**: The data was cleaned and processed to ensure accuracy and consistency.</span>
        - <span class="black-text">**Recommendation Algorithms**: We implemented two types of recommendation algorithms:</span>
          - <span class="black-text">**Content-Based Filtering**: Recommends anime similar to the ones a user likes based on genre and other features.</span>
          - <span class="black-text">**Collaborative-Based Filtering**: Recommends anime based on the preferences of other users who have similar tastes.</span>
        
        ## Future Work
        - <span class="black-text">Improve the collaborative filtering algorithm using a trained model.</span>
        - <span class="black-text">Implement a feedback loop to refine recommendations based on user input.</span>
        - <span class="black-text">Expand the dataset with more comprehensive anime information.</span>
        """, unsafe_allow_html=True)

    elif options == 'Anime Recommendation':
        st.title('Anime Recommendation App')

        # Display an image
        image_path = 'C:/Users/mahla/OneDrive/Desktop/anime.jpg'
        image = Image.open(image_path)
        st.image(image)

        # File upload
        uploaded_file = st.file_uploader("Upload your anime dataset CSV file", type="csv")

        if uploaded_file is not None:
            anime_data = load_data(uploaded_file)

            # User input section
            st.header('Select an algorithm')
            algorithm = st.radio(
                '',
                ('Content Based Filtering', 'Collaborative Based Filtering')
            )

            st.header('Enter Your Three Favorite Movies')
            user_input = []
            user_input.append(st.text_input('Enter first movie choice'))
            user_input.append(st.text_input('Enter second movie choice'))
            user_input.append(st.text_input('Enter third movie choice'))

            # Recommend button
            if st.button('Recommend'):
                if algorithm == 'Content Based Filtering':
                    recommendations = content_based_recommendations(user_input, anime_data)
                else:
                    st.text("Make sure you have uploaded your pickled model before running collaborative filtering!")
                    model_path = 'C:/Users/mahla/Downloads/model/collaborative_model.pkl'
                    recommendations = collaborative_based_recommendations(user_input, model_path, anime_data)

                st.write('Here are your recommendations:')
                for i, rec in enumerate(recommendations):
                    st.write(f"{i + 1}. {rec}")

if __name__ == '__main__':
    main()
