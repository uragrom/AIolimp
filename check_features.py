# Посмотрим топ-слова для каждого кластера
from TeachAItest.check_features import vec


print("Топ-слова по кластерам:")
order_centroids = model_kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vec.get_feature_names_out()

for i in range(7):
    top_words = [terms[ind] for ind in order_centroids[i, :5]]
    print(f"Кластер {i}: {', '.join(top_words)}")