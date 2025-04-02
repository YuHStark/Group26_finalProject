import numpy as np
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import zipfile
import difflib

class RecommendationEngine:
    def __init__(self, processed_data_path):
        """Initialize the lightweight recommendation engine"""
        print(f"Loading data from {processed_data_path}...")
        
        # Handle ZIP file
        if processed_data_path.endswith('.zip'):
            with zipfile.ZipFile(processed_data_path) as zf:
                csv_files = [f for f in zf.namelist() if not f.startswith('__MACOSX')]
                if not csv_files:
                    raise ValueError(f"No CSV files found in {processed_data_path}")
                with zf.open(csv_files[0]) as file:
                    self.df = pd.read_csv(
                        file, 
                        usecols=['book_id', 'book_title', 'book_authors', 'book_desc', 
                                'book_rating', 'book_rating_count', 'book_pages', 'genres']
                    )
        else:
            self.df = pd.read_csv(
                processed_data_path,
                usecols=['book_id', 'book_title', 'book_authors', 'book_desc', 
                        'book_rating', 'book_rating_count', 'book_pages', 'genres']
            )
        
        print(f"Loaded {len(self.df)} books")
        
        # Preprocess text data
        self.df['processed_desc'] = self.df['book_desc'].fillna('').apply(self._preprocess_text)
        
        # Initialize TF-IDF vectorizer and immediately process text
        print("Creating TF-IDF vectors for book descriptions...")
        self.tfidf = TfidfVectorizer(
            max_features=1000,  # Limit feature count
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['processed_desc'])
        print(f"Created TF-IDF matrix with shape: {self.tfidf_matrix.shape}")
        
        # Initialize KNN model (using TF-IDF vectors)
        self._init_knn_model()
        
        # Calculate popularity scores
        self.df['popularity_score'] = (
            self.df['book_rating'] * self.df['book_rating_count']
        ) / (self.df['book_rating_count'] + 10)
    
    def _preprocess_text(self, text):
        """Simple text preprocessing"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _init_knn_model(self):
        """Initialize KNN model for similarity search"""
        print("Initializing KNN model...")
        self.knn_model = NearestNeighbors(
            n_neighbors=10,  # A few extra to prevent self-matching
            metric='cosine',
            algorithm='brute'  # Brute is usually faster for high-dimensional sparse data
        )
        self.knn_model.fit(self.tfidf_matrix)
        print("KNN model initialized")
    
    def content_based_filter(self, genre=None, style=None, min_rating=0, max_pages=None):
        """Content-based filtering"""
        filtered_df = self.df.copy()
        
        # Filter by genre
        if genre and genre != "":
            genre_pattern = re.compile(genre, re.IGNORECASE)
            filtered_df = filtered_df[filtered_df['genres'].apply(
                lambda x: bool(genre_pattern.search(str(x))))]
        
        # Filter by rating
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['book_rating'] >= min_rating]
        
        # Filter by page count
        if max_pages and max_pages > 0:
            filtered_df = filtered_df[filtered_df['book_pages'] <= max_pages]
        
        # Filter by style (search for keywords in description)
        if style and style != "":
            style_pattern = re.compile(style, re.IGNORECASE)
            filtered_df = filtered_df[filtered_df['book_desc'].apply(
                lambda x: bool(style_pattern.search(str(x))))]
        
        # Sort by rating
        filtered_df = filtered_df.sort_values(by='book_rating', ascending=False)
        
        return filtered_df.head(5)
    
    def popularity_rank_recommend(self, min_rating=0, max_pages=None, top_n=5):
        """Popularity-based recommendation"""
        filtered_df = self.df.copy()
        
        # Apply filters
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['book_rating'] >= min_rating]
        
        if max_pages and max_pages > 0:
            filtered_df = filtered_df[filtered_df['book_pages'] <= max_pages]
        
        # Sort by popularity score
        filtered_df = filtered_df.sort_values(by='popularity_score', ascending=False)
        
        return filtered_df.head(top_n)
    
    def find_similar_books_knn(self, book_title, n=3):
        """Find similar books using TF-IDF vectors"""
        # Find book index
        book_indices = self.df[self.df['book_title'].str.contains(book_title, case=False, na=False)].index
        
        if len(book_indices) == 0:
            return pd.DataFrame()  # Book not found
        
        book_idx = book_indices[0]  # Take the first match
        
        # Get TF-IDF vector for the query book
        book_tfidf = self.tfidf_matrix[book_idx]
        
        # Use KNN to find nearest neighbors
        distances, indices = self.knn_model.kneighbors(book_tfidf)
        
        # Skip the first result as it's the query book itself
        similar_books = self.df.iloc[indices.flatten()[1:n+1]]
        
        return similar_books
    
    def ensemble_recommendations(self, genre=None, style=None, min_rating=0, max_pages=None, top_n=5):
        """Combine multiple recommendation algorithms for better results"""
        # Get recommendations from each algorithm
        content_recs = self.content_based_filter(genre, style, min_rating, max_pages)
        popularity_recs = self.popularity_rank_recommend(min_rating, max_pages)
        
        # Combine and deduplicate recommendations
        all_books = pd.concat([
            content_recs.assign(algorithm='content'),
            popularity_recs.assign(algorithm='popularity')
        ])
        
        # Count how many times each book appears across algorithms
        book_counts = all_books.groupby('book_id').size().reset_index(name='algorithm_count')
        
        # Merge with original recommendations to get all metadata
        scored_recs = all_books.drop_duplicates(subset='book_id').merge(
            book_counts, on='book_id', how='left')
        
        # Sort first by count (how many algorithms recommended it), then by rating
        final_recs = scored_recs.sort_values(
            by=['algorithm_count', 'book_rating', 'popularity_score'], 
            ascending=[False, False, False]
        )
        
        return final_recs.head(top_n)[['book_id', 'book_title', 'book_authors', 
                                      'book_rating', 'book_pages', 'genres']]
    
    def get_book_details(self, book_title):
        """Get detailed information for a specific book"""
        target = book_title.lower().strip()
        exact_matches = self.df[self.df['book_title'].str.lower().str.strip() == target]
        if not exact_matches.empty:
            book = exact_matches.iloc[0]
       else:
            all_titles = self.df['book_title'].tolist()
            close_matches = difflib.get_close_matches(book_title, all_titles, n=1, cutoff=0.6)
            if close_matches:
                best_match = close_matches[0]
                fuzzy_matches = self.df[self.df['book_title'] == best_match]
                if not fuzzy_matches.empty:
                    book = fuzzy_matches.iloc[0]
                else:
                    return None
            else:
                return None
        
        
        details = {
            'title': book['book_title'],
            'author': book['book_authors'],
            'description': book['book_desc'],
            'genres': book['genres'],
            'rating': book['book_rating'],
            'rating_count': book['book_rating_count'],
            'pages': book['book_pages'],
            'format': book.get('book_format', 'Unknown')
        }
        
        return details
