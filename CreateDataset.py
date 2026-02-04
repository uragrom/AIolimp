import pandas as pd, glob, re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. Сбор данных
data = []
for f in glob.glob('train/**/*.html', recursive=True):
    html = open(f, encoding='utf-8').read()
    text = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
    data.append({'file': f, 'raw_text': text})

df = pd.DataFrame(data)

# 2. Очистка
df['clean'] = df['raw_text'].str.lower().apply(lambda x: re.sub(r'[^a-zа-я\s]', ' ', str(x)))

# 3. Кластеризация
vec = TfidfVectorizer(max_features=2000, stop_words='english')
X = vec.fit_transform(df['clean'])

model_kmeans = KMeans(n_clusters=7, random_state=42, n_init='auto')
df['target_cluster'] = model_kmeans.fit_predict(X)

# ВЫВОД ТОП-СЛОВ (Для интерпретации в отчете)
print("\n--- ТОП-СЛОВА ПО КЛАСТЕРАМ ---")
order_centroids = model_kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vec.get_feature_names_out()

for i in range(7):
    top_words = [terms[ind] for ind in order_centroids[i, :5]]
    print(f"Кластер {i}: {', '.join(top_words)}")

# 4. Сохранение
df.to_csv('processed_train.csv', index=False)
print("\nDataSet готов для Модуля Б!")