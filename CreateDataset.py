import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. ЗАГРУЗКА
train = pd.read_csv('dataset.csv') # Убедись, что имена файлов правильные
test = pd.read_csv('dataset.csv')

# --- ДИНАМИЧЕСКИЙ ПОИСК КОЛОНОК ---
# Ищем колонки, в названии которых есть 'title' или 'abstract' (без учета регистра)
def get_col(df, keyword):
    cols = [c for c in df.columns if keyword in c.lower()]
    return cols[0] if cols else None

t_col_train = get_col(train, 'title')
a_col_train = get_col(train, 'abstract')
t_col_test = get_col(test, 'title')
a_col_test = get_col(test, 'abstract')

# 2. ОЧИСТКА
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zа-я\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Объединяем найденные колонки. Если какой-то нет — используем пустую строку.
train['clean'] = (train[t_col_train].fillna('') + " " + train[a_col_train].fillna('')).apply(clean)
test['clean'] = (test[t_col_test].fillna('') + " " + test[a_col_test].fillna('')).apply(clean)

# 3. ВЕКТОРИЗАЦИЯ
# Обучаем (fit) на TRAIN, применяем (transform) к обоим
vec = TfidfVectorizer(max_features=2000, stop_words='english')
X_train = vec.fit_transform(train['clean'])
X_test = vec.transform(test['clean'])

# 4. КЛАСТЕРИЗАЦИЯ (Только для TRAIN)
model = KMeans(n_clusters=7, random_state=42, n_init='auto')
train['target_cluster'] = model.fit_predict(X_train)

# 5. ВЫВОД (Сохраняем всё, что получилось)
train.to_csv('processed_train.csv', index=False)
test.to_csv('processed_test.csv', index=False)

print("Готово! Колонки найдены автоматически, файлы созданы.")