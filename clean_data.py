import pandas as pd

# Load final dataset
final_dataset = pd.read_csv('final_dataset.csv')

# 1️⃣ Check initial info
print("Before cleaning:")
print(final_dataset.info())
print(final_dataset.isnull().sum())

# 2️⃣ Remove duplicates if any
final_dataset.drop_duplicates(inplace=True)

# 3️⃣ Drop rows with missing crucial data
final_dataset.dropna(subset=['movieId', 'title', 'genres'], inplace=True)

# 4️⃣ Fill missing tags with empty string
final_dataset['tag'] = final_dataset['tag'].fillna('')

# 5️⃣ If 'rating' is missing, you can either drop those or fill with average (optional)
final_dataset['rating'] = final_dataset['rating'].fillna(final_dataset['rating'].mean())

final_dataset = final_dataset[final_dataset['tmdbId'].notna()]

# 6️⃣ Check again after cleaning
print("\nAfter cleaning:")
print(final_dataset.info())
print(final_dataset.isnull().sum())

# 7️⃣ Save the cleaned final dataset
final_dataset.to_csv('final_dataset_cleaned.csv', index=False)

print("\n✅ final_dataset_cleaned.csv is ready!")
