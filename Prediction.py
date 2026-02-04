import pandas as pd, glob, re, joblib
from bs4 import BeautifulSoup

# 1. Загрузка "мозгов"
model = joblib.load('model.pkl')
vec = joblib.load('vec.pkl')

# 2. Чтение папки test
files = glob.glob('test/**/*.html', recursive=True)
if files:
    texts = []
    for f in files:
        txt = BeautifulSoup(open(f, encoding='utf-8').read(), 'html.parser').get_text()
        texts.append(re.sub(r'[^a-zа-я\s]', ' ', txt.lower()))

    # 3. Работа модели
    X_test = vec.transform(texts)
    preds = model.predict(X_test)

    # 4. Результат
    pd.DataFrame({'file': files, 'cluster': preds}).to_csv('test_results.csv', index=False)
    print("Готово!")