import pandas as pd, glob, re, joblib, charset_normalizer
from bs4 import BeautifulSoup

m, v = joblib.load('model.pkl'), joblib.load('vec.pkl')

files = glob.glob('test/**/*.html', recursive=True)

def smart_read(f):
    raw = open(f, 'rb').read()
    result = charset_normalizer.from_bytes(raw).best()
    return str(result) if result else raw.decode('utf-8', errors='ignore')

data = []
for f in files:
    txt = BeautifulSoup(smart_read(f), 'html.parser').get_text()
    data.append(re.sub(r'[^a-zа-яё\s]', ' ', txt.lower()))

preds = m.predict(v.transform(data))
pd.DataFrame({'file': files, 'cluster': preds}).to_csv('test_results.csv', index=False)

names = v.get_feature_names_out()
for i, coefs in enumerate(m.coefs_[0].T[:7]): # смотрим первые 7 кластеров
    top_indices = coefs.argsort()[-10:]
    print(f"Кластер {i}: {', '.join(names[j] for j in top_indices)}")

print(f"Готово!")