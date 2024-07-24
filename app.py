import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Content-Based Filtering
def content_based_recommender(user_input, data):
    data['combined_features'] = data.apply(lambda row: f"{row['genre']} {row['name']}", axis=1)
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)

    def get_title_from_index(index):
        return data.loc[index, 'name']

    def get_index_from_title(title):
        try:
            return data[data['name'] == title].index[0]
        except IndexError:
            return None

    recommendations = []
    for title in user_input:
        index = get_index_from_title(title)
        if index is not None:
            similar_items = list(enumerate(cosine_sim[index]))
            sorted_similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:11]
            recommendations.extend([get_title_from_index(i[0]) for i in sorted_similar_items])
        else:
            recommendations.append(f"{title} not found in dataset")

    return list(set(recommendations))

# Collaborative-Based Filtering
def collaborative_based_recommender(user_input, data):
    data['rating'] = data['rating'].fillna(0)
    data = data.astype({'anime_id': 'int64'})
    data = data.drop_duplicates(['anime_id', 'rating'])
    user_item_matrix = data.pivot_table(index='anime_id', columns='name', values='rating').fillna(0)

    user_input_indices = [data[data['name'] == title].index[0] for title in user_input if title in data['name'].values]
    user_vector = np.zeros(len(user_item_matrix.columns))
    for idx in user_input_indices:
        user_vector[idx] = 10

    similarity_scores = cosine_similarity([user_vector], user_item_matrix.T)[0]
    similar_items = list(enumerate(similarity_scores))
    sorted_similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:11]

    recommendations = [user_item_matrix.columns[i] for i, score in sorted_similar_items if score > 0]

    return recommendations if recommendations else ["No suggestion possible!"]

# Main Streamlit App
def main():
    st.markdown("<h1 style='color: #FF6347;'>Anime Recommendation App</h1>", unsafe_allow_html=True)

    # Display an image
    image_path = 'anime.jpg'
    image = Image.open(image_path)
    st.image(image)

    # Sidebar with app description
    st.sidebar.markdown("<h2 style='color: #4682B4;'>About the App</h2>", unsafe_allow_html=True)
    st.sidebar.write("""
    This Anime Recommendation System uses both Content-Based and Collaborative-Based Filtering algorithms to suggest anime to users.
    - **Content-Based Filtering**: Recommends anime similar to the ones you like based on genre and other features.
    - **Collaborative-Based Filtering**: Recommends anime based on the preferences of other users who have similar tastes.
    
    Simply choose the algorithm, enter your three favorite anime, and get your recommendations!
    """)

    # Customize sidebar and main content background color
    st.sidebar.markdown(
        """
        <style>
        .css-1d391kg {background-color: #F0F8FF;}
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .stApp {background-color: #FAFAD2;}
        </style>
        """, 
        unsafe_allow_html=True
    )

    # File upload
    uploaded_file = st.file_uploader("Upload your anime dataset CSV file", type="csv")

    if uploaded_file is not None:
        anime_data = load_data(uploaded_file)

        # User input section with colored headers
        st.markdown("<h2 style='color: #4682B4;'>Select an algorithm</h2>", unsafe_allow_html=True)
        algorithm = st.radio(
            '',
            ('Content Based Filtering', 'Collaborative Based Filtering')
        )

        st.markdown("<h2 style='color: #4682B4;'>Enter Your Three Favorite Movies</h2>", unsafe_allow_html=True)
        user_input = []
        user_input.append(st.text_input('Enter first movie choice'))
        user_input.append(st.text_input('Enter second movie choice'))
        user_input.append(st.text_input('Enter third movie choice'))

        # Recommend button with loading spinner
        if st.button('Recommend'):
            with st.spinner('Generating recommendations...'):
                if algorithm == 'Content Based Filtering':
                    recommendations = content_based_recommender(user_input, anime_data)
                else:
                    recommendations = collaborative_based_recommender(user_input, anime_data)

            st.write('Here are your recommendations:')
            for i, rec in enumerate(recommendations):
                st.write(f"{i + 1}. {rec}")

if __name__ == '__main__':
    main()
