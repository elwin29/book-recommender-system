import pickle
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.DEBUG)

st.header('Book Recommender System Using Hybrid Machine Learning')

logging.info("Loading models and data...")
@st.cache_resource
def load_model():
    return pickle.load(open('artifacts/model.pkl', 'rb'))

@st.cache_data
def load_data():
    book_names = pickle.load(open('artifacts/books_name.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
    book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
    book_embeddings = pickle.load(open('artifacts/book_embeddings.pkl', 'rb'))
    books = pickle.load(open('artifacts/books.pkl', 'rb'))
    return book_names, final_rating, book_pivot, book_embeddings, books

model = load_model()
book_names, final_rating, book_pivot, book_embeddings, books = load_data()
logging.info("Models and data loaded successfully.")

def fetch_poster(suggestion):
    try:
        book_name = []
        poster_url = []

        for book_title, _ in suggestion:
            book_name.append(book_title)

        for name in book_name:
            ids = books[books['title'] == name].index
            if not ids.empty:
                url = books.loc[ids[0]]['image']
                poster_url.append(url)
            else:
                poster_url.append('')  # Append empty string if no URL found

        return poster_url
    except Exception as e:
        logging.error(f"Error in fetch_poster: {e}")
        return []

def collaborative_recommendations(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distances, indices = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
        suggestions = [(book_pivot.index[i], 'Collaborative Filtering') for i in indices.flatten() if book_pivot.index[i] != book_name]
        return suggestions
    except Exception as e:
        logging.error(f"Error in collaborative_recommendations: {e}")
        return []

def content_based_recommendations_bert(book_title, top_n=10):
    try:
        index = books[books['title'] == book_title].index[0]
        query_embedding = book_embeddings[index].unsqueeze(0)  # Shape (1, embedding_dim)

        similarities = cosine_similarity(query_embedding.cpu().numpy(), book_embeddings.cpu().numpy()).flatten()
        similar_indices = similarities.argsort()[-top_n-1:-1][::-1]  # Exclude the book itself
        recommended_books = [(books.iloc[similar_indices]['title'].iloc[i], 'Content-based Filtering') for i in range(top_n)]

        return recommended_books
    except Exception as e:
        logging.error(f"Error in generating recommendations: {str(e)}")
        return []

def hybrid_recommendations(book_name):
    try:
        if book_name in book_pivot.index and (final_rating[final_rating['title'] == book_name].iloc[0, -1] > 140):
            collab_recs = collaborative_recommendations(book_name)
        else:
            collab_recs = []

        content_recs = content_based_recommendations_bert(book_name)

        combined_recs = list(collab_recs)
        for rec in content_recs:
            if rec[0] not in [book[0] for book in combined_recs]:
                combined_recs.append(rec)
        
        return combined_recs[:10]
    except Exception as e:
        logging.error(f"Error in hybrid_recommendations: {e}")
        return []

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    try:
        recommended_books = hybrid_recommendations(selected_books)
        poster_url = fetch_poster(recommended_books)

        if recommended_books and poster_url:
            for i in range(0, len(recommended_books), 5):
                cols = st.columns(5)
                for idx, col in enumerate(cols):
                    if i + idx < len(recommended_books):
                        book_title, rec_type = recommended_books[i + idx]
                        with col:
                            st.text(f"{book_title}")
                            st.markdown(f"</p><p style='font-size:12px; color:grey;'>{rec_type}</p>", unsafe_allow_html=True)
                            if poster_url[i + idx]:  # Only show image if URL is valid
                                st.image(poster_url[i + idx])
    except Exception as e:
        logging.error(f"Error in showing recommendations: {e}")
        st.error("An error occurred while generating recommendations. Please try again.")
