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
def collaborative_based_recommendations(user_input, data):
    data['rating'] = data['rating'].fillna(0)
    data = data.astype({'anime_id': 'int64'})
    data = data.drop_duplicates(['anime_id', 'rating'])
    data_pivot = data.pivot_table(index='anime_id', columns='name', values='rating').fillna(0)
    user_input = [i for i in user_input]
    a = np.zeros(len(data_pivot.columns))
    for i in user_input:
        a[i - 1] = 10
    cos_sim = cosine_similarity([a], data_pivot)
    df_cos_sim = pd.DataFrame(cos_sim[0], index=data_pivot.index)
    df_cos_sim.columns = ['Cosine Similarity']
    df_cos_sim = df_cos_sim.sort_values(by='Cosine Similarity', ascending=False)
    if df_cos_sim.iloc[1, 0] < 0.4:
        return "No suggestion possible!"
    idx = df_cos_sim.iloc[1, :].name
    reco = []
    for i in data_pivot.columns:
        if data_pivot.loc[idx, i] == 10:
            reco.append(data_pivot.columns[i - 1])
    return reco

# Main Streamlit App
def main():
    st.markdown("<h1 style='color: #FF6347;'>Anime Recommendation App</h1>", unsafe_allow_html=True)

    # Display an image
    image_path = 'C:/Users/mahla/OneDrive/Desktop/anime.jpg'
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
                    recommendations = content_based_recommendations(user_input, anime_data)
                else:
                    recommendations = collaborative_based_recommendations(user_input, anime_data)

            st.write('Here are your recommendations:')
            for i, rec in enumerate(recommendations):
                st.write(f"{i + 1}. {rec}")

if __name__ == '__main__':
    main()
