# 📚 Book Recommendation System

## Project Overview

This project aims to build a book recommendation system using collaborative filtering and content-based filtering techniques. The system recommends books to users based on their past ratings and the content of the books.

## Dataset

The dataset used for this project is from Kaggle and includes user ratings for books. It contains information such as user IDs, book ISBNs, and ratings. Additionally, a books dataset with details like title, author, year of publication, and publisher is used for content-based filtering.

🔗 [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/ra4u12/bookrecommendation)

## Installation

To install the required packages, run:

    pip install -r requirements.txt

## Usage

1. **Preprocess the data**:
   The data preprocessing is performed in the `book-recommender.ipynb` notebook. This includes filtering users, creating user-item matrices, and calculating similarities.

2. **Run the application**:
   The `app.py` file contains the Flask application to serve the recommendation system. To run the app:
   
    streamlit run app.py

## Methodology

### Data Preprocessing

- **Filtering Users**:
  Users who have rated more than 200 books are selected to ensure a dense user-item matrix.
  
- **Creating User-Item Matrix**:
  A pivot table is created with users as rows, books as columns, and ratings as values.

- **Calculating Similarities**:
  Cosine similarity is used to calculate the similarity between users based on their ratings.

### Recommendation Techniques

1. **User-Based Collaborative Filtering**:
   Recommendations are made by finding similar users and suggesting books that those users have highly rated.

2. **Content-Based Filtering**:
   Book embeddings are created using a pre-trained BERT model. Similar books are recommended based on cosine similarity of these embeddings.

3. **Hybrid Approach**:
   Combines collaborative and content-based filtering to provide more robust recommendations.

### Saving Models

The trained models and data structures are saved for later use:
- `book_embeddings.pkl`: Book embeddings for content-based filtering.
- `model.pkl`: Collaborative filtering model.
- `books_name.pkl`: Index of book names.
- `book_pivot.pkl`: User-item matrix.
- `books.pkl`: Books dataset.
- `final_rating.pkl`: Final user ratings.

## Results

The recommendation system provides book recommendations based on user preferences and book content. Users can receive hybrid recommendations that leverage both collaborative and content-based techniques.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
