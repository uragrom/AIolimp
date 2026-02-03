import pandas as pd, matplotlib.pyplot as plt
import re, nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words_list = list(stopwords.words('russian')) + list(stopwords.words('english'))

df = pd.read_csv('dataset.csv')

col = [c for c in df.columns if 'abstract' in c.lower() or 'text' in c.lower()][0]
texts = df[col].fillna('').str.lower().apply(lambda x: re.sub(r'[^a-zа-я\s]', ' ', x))

X = TfidfVectorizer(max_features=2000, stop_words=stop_words_list).fit_transform(texts)

distortions = [KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X).inertia_ for k in range(2, 12)]

plt.plot(range(2, 12), distortions, 'bx-')
plt.xlabel('Количество кластеров (k)'); plt.ylabel('Инерция'); plt.title('Метод локтя (RU+EN)')
plt.show()