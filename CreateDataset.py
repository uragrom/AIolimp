import pandas as pd, glob, re, joblib, nltk, charset_normalizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

def smart_read(f):
    raw = open(f, 'rb').read()
    result = charset_normalizer.from_bytes(raw).best()
    return str(result) if result else raw.decode('utf-8', errors='ignore')

data = []
for f in glob.glob('train/**/*.html', recursive=True):
    txt = BeautifulSoup(smart_read(f), 'html.parser').get_text()
    data.append(re.sub(r'[^a-zа-яё\s]', ' ', txt.lower()))

nltk.download('stopwords')
stop_words_list = stopwords.words('english') + stopwords.words('russian')

vec = TfidfVectorizer(max_features=1000, stop_words=stop_words_list)
X = vec.fit_transform(data)
labels = KMeans(n_clusters=7, random_state=42, n_init='auto').fit_predict(X)

model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    max_iter=100,
    verbose=True,
    early_stopping=False,
    random_state=42
)
model.fit(X, labels)

joblib.dump(model, 'model.pkl')
joblib.dump(vec, 'vec.pkl')
print("Готово! Русский язык в любой кодировке обработан.")