import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('dataset.csv')
X = TfidfVectorizer(max_features=1000, stop_words='english').fit_transform(df['patent_abstract'].fillna(''))

distortions = []
K = range(2, 12)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Инерция')
plt.title('Метод локтя для поиска оптимального K')
plt.show()