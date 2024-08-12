import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from surprise import Dataset, Reader # type: ignore
import pickle
from PIL import Image # type: ignore

# Load data
@st.cache_data
def load_anime_data():
    return pd.read_csv('anime.csv')  # Adjust the path to your dataset

@st.cache_data
def load_train_data():
    return pd.read_csv('train.csv')  # Adjust the path to your dataset

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
    for anime in user_input:
        try:
            anime_index = get_index_from_title(anime)
            similar_anime = list(enumerate(cosine_sim[anime_index]))
            sorted_similar_anime = sorted(similar_anime, key=lambda x: x[1], reverse=True)
            for element in sorted_similar_anime[1:11]:
                recommendations.append(get_title_from_index(element[0]))
        except IndexError:
            recommendations.append("Anime not found in dataset")
    return recommendations

# Collaborative-Based Filtering
def collaborative_based_recommendations(user_input, train_data, anime_data, model_path):
    # Load pickled model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Prepare train data
    train_data = train_data[['anime_id', 'user_id', 'rating']]
    train_data['rating'] = train_data['rating'].fillna(0)
    train_data = train_data.drop_duplicates(['anime_id', 'user_id'])

    # Convert to surprise Dataset
    reader = Reader(rating_scale=(0, 10))
    surprise_data = Dataset.load_from_df(train_data[['user_id', 'anime_id', 'rating']], reader)
    trainset = surprise_data.build_full_trainset()

    # Rebuild the model using the full dataset
    model.fit(trainset)

    # Prepare user input
    user_input_ids = [anime_data[anime_data['name'] == title]['anime_id'].values[0] for title in user_input if title in anime_data['name'].values]

    # Make predictions for all anime in the dataset
    predictions = []
    for anime_id in anime_data['anime_id'].unique():
        pred = model.predict(uid='user', iid=anime_id)
        predictions.append((anime_id, pred.est))

    # Get recommendations
    recommendation_scores = sorted(predictions, key=lambda x: x[1], reverse=True)
    recommended_anime_ids = [i[0] for i in recommendation_scores[:10]]

    # Merge with anime data to get names
    recommended_anime = anime_data[anime_data['anime_id'].isin(recommended_anime_ids)]['name'].tolist()

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
    options = st.sidebar.radio("Select a page:", ['Team Page', 'Project Overview', 'Anime Recommendation', 'Feedback'])

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

        # Load data
        anime_data = load_anime_data()
        train_data = load_train_data()

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
                # Path to pickled model
                model_path = 'C:/Users/mahla/Downloads/model/collaborative_model.pkl'  # Use the correct absolute path
                recommendations = collaborative_based_recommendations(user_input, train_data, anime_data, model_path)

            st.write('Here are your recommendations:')
            for i, rec in enumerate(recommendations):
                st.write(f"{i + 1}. {rec}")

    elif options == 'Feedback':
        st.title("Feedback Page")
        st.markdown("""
        ## We Value Your Feedback!
        Please let us know what you think about our Anime Recommendation App. Your feedback is important to us and helps us improve our service.

        ### Feedback Form
        """)
        
        name = st.text_input("Name")
        email = st.text_input("Email")
        feedback = st.text_area("Your Feedback")
        
        if st.button("Submit"):
            st.write("Thank you for your feedback!")
            # Here you can add code to save the feedback to a database or send it via email

if __name__ == '__main__':
    main()
