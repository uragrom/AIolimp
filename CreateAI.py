import pandas as pd, glob, re, joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

# 1. Сбор и очистка
data = []
for f in glob.glob('train/**/*.html', recursive=True):
    txt = BeautifulSoup(open(f, encoding='utf-8').read(), 'html.parser').get_text()
    data.append(re.sub(r'[^a-zа-я\s]', ' ', txt.lower()))

# 2. Векторизация (Вместо Tokenizer)
vec = TfidfVectorizer(max_features=1000, stop_words='english')
X = vec.fit_transform(data)

# 3. Кластеризация (Модуль А)
labels = KMeans(n_clusters=7, random_state=42, n_init='auto').fit_predict(X)

# 4. Обучение нейросети (MLP из sklearn — это и есть нейросеть, но без Keras)
model = MLPClassifier(
hidden_layer_sizes=(256, 128),
    max_iter=100,            # Можно поставить побольше, автостоп всё равно сработает раньше
    verbose=True,
    early_stopping=True,     # ВКЛЮЧАЕМ АВТОСТОП
    validation_fraction=0.1, # 10% данных для валидации
    n_iter_no_change=5,      # Терпение модели (5 эпох без улучшений)
    random_state=42
)
model.fit(X, labels)

# 5. Сохранение (Всего 2 маленьких файла)
joblib.dump(model, 'model.pkl')
joblib.dump(vec, 'vec.pkl')
print("Обучено!")