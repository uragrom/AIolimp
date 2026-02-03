import pandas as pd, re, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

train, test = pd.read_csv('dataset.csv'), pd.read_csv('dataset.csv')

try:
    nltk.download('stopwords')
    stop_words_list = list(stopwords.words('russian')) + list(stopwords.words('english'))
except:
    stop_words_list = 'english'

get_col = lambda df, k: [c for c in df.columns if k in c.lower()][0]
def clean(t): return re.sub(r'\s+', ' ', re.sub(r'[^a-zа-я\s]', ' ', str(t).lower())).strip()

for df in [train, test]:
    t, a = get_col(df, 'title'), get_col(df, 'abstract')
    df['clean'] = (df[t].fillna('') + " " + df[a].fillna('')).apply(clean)

vec = TfidfVectorizer(max_features=2000, stop_words=stop_words_list)
X_train = vec.fit_transform(train['clean'])
X_test = vec.transform(test['clean'])

model = KMeans(n_clusters=7, random_state=42, n_init='auto')
train['target_cluster'] = model.fit_predict(X_train)

train.to_csv('processed_train.csv', index=False)
test.to_csv('processed_test.csv', index=False)
print("Готово! Данные очищены, стоп-слова RU+EN применены, кластеры созданы.")