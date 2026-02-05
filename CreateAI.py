import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('processed_train.csv')
texts = df['clean'].astype(str)
labels = df['target_cluster']

vec = TfidfVectorizer(max_features=1000, stop_words='english')
X = vec.fit_transform(texts)

model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100,
                      verbose=True, early_stopping=True, random_state=42)
model.fit(X, labels)

joblib.dump(model, 'model.pkl')
joblib.dump(vec, 'vec.pkl')
print("Нейросеть обучена на данных из CSV!")