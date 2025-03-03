from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load dataset
df = pd.read_csv('final_dataset_cleaned.csv')

# Preprocess dataset
df['genres'] = df['genres'].fillna('')
df['tag'] = df['tag'].fillna('')
df['tags'] = (df['genres'] + ' ' + df['tag']).str.lower()
df = df.drop_duplicates(subset='title').reset_index(drop=True)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# Nearest Neighbors model
model = NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').lower()
    matches = df[df['title'].str.lower().str.contains(query, na=False)]['title'].head(10).tolist()
    return jsonify(matches)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_name = data.get('movie', '').strip()

    matched_movies = df[df['title'] == movie_name]

    if matched_movies.empty:
        return jsonify([])

    idx = matched_movies.index[0]
    movie_vector = tfidf_matrix[idx]
    distances, indices = model.kneighbors(movie_vector)

    recommended_titles = df.iloc[indices[0]].title.tolist()

    if movie_name in recommended_titles:
        recommended_titles.remove(movie_name)

    return jsonify(recommended_titles)

if __name__ == '__main__':
    app.run(debug=True)
