# --- 1. ПОДГОТОВКА СТОП-СЛОВ ---
try:
    # Пытаемся взять системные стоп-слова
    from nltk.corpus import stopwords
    stop_words = list(stopwords.words('russian')) + list(stopwords.words('english'))
except Exception:
    # Если словари не скачаны заранее, используем пустой список (чтобы код не упал)
    print("ВНИМАНИЕ: Словари NLTK не найдены, используем только custom_stops")
    stop_words = []

# Добавляем наш список, который мы составили вручную (он всегда под рукой)
custom_stops = ['patent', 'method', 'system', 'device', 'said', 'comprising', 'fig',
                'изобретение', 'способ', 'устройство', 'фиг', 'данный']
stop_words.extend(custom_stops)