import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('dataset.csv')
vec = TfidfVectorizer(max_features=2000, stop_words='english')
X = vec.fit_transform(df['patent_abstract'].fillna(''))

# Смотрим самые важные слова
features = vec.get_feature_names_out()
importance = X.sum(axis=0).A1
vocabulary = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)

print("Топ-10 значимых слов в датасете:")
for word, score in vocabulary[:10]:
    print(f"{word}: {score:.2f}")