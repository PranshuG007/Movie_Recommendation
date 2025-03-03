import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset
final_dataset_cleaned = pd.read_csv('final_dataset_cleaned.csv')

# Check and create 'tags' column if it doesn't exist
if 'tags' not in final_dataset_cleaned.columns:
    final_dataset_cleaned['genres'] = final_dataset_cleaned['genres'].fillna('')
    final_dataset_cleaned['tag'] = final_dataset_cleaned['tag'].fillna('')
    final_dataset_cleaned['tags'] = final_dataset_cleaned['genres'] + ' ' + final_dataset_cleaned['tag']

# Reset index
final_dataset_cleaned = final_dataset_cleaned.reset_index(drop=True)

# Remove duplicates, keeping only the first occurrence of each movie title
final_dataset_cleaned = final_dataset_cleaned.drop_duplicates(subset='title').reset_index(drop=True)

print(f"Number of unique movies: {final_dataset_cleaned['title'].nunique()}")
print(f"Total movies after removing duplicates: {len(final_dataset_cleaned)}")

# Vectorize 'tags'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(final_dataset_cleaned['tags'])

# Use NearestNeighbors to find similar movies
model = NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(tfidf_matrix)

# Recommendation function
def recommend(title):
    if title not in final_dataset_cleaned['title'].values:
        return f"❌ Movie '{title}' not found in the dataset."
    
    idx = final_dataset_cleaned[final_dataset_cleaned['title'] == title].index[0]
    movie_vector = tfidf_matrix[idx]
    
    distances, indices = model.kneighbors(movie_vector)
    
    recommended_titles = final_dataset_cleaned.iloc[indices[0]].title.tolist()
    recommended_titles.remove(title)  # Remove the input movie itself if present
    return recommended_titles

# Test recommendation
movie_name = 'Toy Story (1995)'
recommendations = recommend(movie_name)
print(f"✅ Optimized Recommendations for '{movie_name}':")
print(recommendations)
print(final_dataset_cleaned[final_dataset_cleaned['title'] == 'Toy Story (1995)'])
print(f"Number of unique movies: {final_dataset_cleaned['title'].nunique()}")
print(f"Total movies: {len(final_dataset_cleaned)}")
