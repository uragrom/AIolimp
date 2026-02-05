import pandas as pd, glob, re, joblib
from bs4 import BeautifulSoup

m, v = joblib.load('model.pkl'), joblib.load('vec.pkl')

files = glob.glob('test/**/*.html', recursive=True)
data = []
for f in files:
    t = BeautifulSoup(open(f, encoding='utf-8', errors='ignore').read(), 'html.parser').get_text()
    data.append(re.sub(r'[^a-zа-яё\s]', ' ', t.lower()))

preds = m.predict(v.transform(data))
pd.DataFrame({'file': files, 'cluster': preds}).to_csv('test_results.csv', index=False)

names = v.get_feature_names_out()
for i, coefs in enumerate(m.coefs_[0].T[:7]): # смотрим первые 7 кластеров
    top_indices = coefs.argsort()[-10:]
    print(f"Кластер {i}: {', '.join(names[j] for j in top_indices)}")

print(f"Готово!")