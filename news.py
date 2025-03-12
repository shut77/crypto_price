import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
import pandas as pd
import time
from fake_useragent import UserAgent
import feedparser  # Для парсинга RSS-каналов

# Загрузка ресурсов NLTK
nltk.download('vader_lexicon')
nltk.download('punkt')

# Инициализация анализатора настроений
analyzer = SentimentIntensityAnalyzer()

# Список криптовалют для анализа (можно расширять динамически)
CRYPTO_LIST = ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana', 'polkadot', 'dogecoin', 'litecoin', "melania"]

# Список сайтов для парсинга
NEWS_SOURCES = [
    'https://coinmarketcap.com/headlines/news/',
    'https://cryptonews.com/',
    'https://www.cointelegraph.com/',
    'https://news.bitcoin.com/feed/',  # RSS-канал
    'https://cryptopotato.com/feed/'  # RSS-канал
]

# Инициализация UserAgent
ua = UserAgent()


def fetch_news(url, retries=3):
    """Получение новостей с сайта или RSS-канала"""
    for attempt in range(retries):
        try:
            if url.endswith('/feed/'):  # Если это RSS-канал
                feed = feedparser.parse(url)
                headlines = [entry.title for entry in feed.entries]
            else:  # Если это обычный сайт
                headers = {
                    'User-Agent': ua.random
                }
                response = requests.get(url, headers=headers, timeout=60)
                soup = BeautifulSoup(response.text, 'html.parser')

                if 'coinmarketcap' in url:
                    headlines = [item.text.strip() for item in soup.find_all('a', class_='cmc-link') if
                                 'news' in item.get('href', '')]
                else:
                    headlines = [item.text.strip() for item in soup.find_all('a') if
                                 any(crypto in item.text.lower() for crypto in CRYPTO_LIST)]

            return headlines
        except requests.exceptions.RequestException as e:
            print(f"Попытка {attempt + 1} из {retries} не удалась: {e}")
            time.sleep(10)  # Пауза перед повторной попыткой
    return []  # Возвращаем пустой список, если все попытки неудачны


def analyze_sentiment(text):
    """Анализ настроения текста"""
    return analyzer.polarity_scores(text)['compound']


def extract_crypto_mentions(text):
    """Извлечение упоминаний криптовалют в тексте"""
    return [crypto for crypto in CRYPTO_LIST if crypto in text.lower()]


def update_crypto_list():
    """Обновление списка криптовалют (например, из CoinMarketCap)"""
    try:
        response = requests.get('https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing', timeout=30)
        data = response.json()
        new_cryptos = [crypto['slug'] for crypto in data['data']['cryptoCurrencyList']]
        return list(set(CRYPTO_LIST + new_cryptos))  # Объединяем старый и новый список
    except Exception as e:
        print(f"Ошибка при обновлении списка криптовалют: {e}")
        return CRYPTO_LIST  # Возвращаем старый список в случае ошибки


def main():
    # Обновляем список криптовалют
    global CRYPTO_LIST
    CRYPTO_LIST = update_crypto_list()
    print(f"Анализируемые криптовалюты: {CRYPTO_LIST}")

    # Словарь для хранения результатов
    crypto_sentiment = defaultdict(list)

    # Парсинг новостей с каждого сайта
    for source in NEWS_SOURCES:
        print(f"Парсинг новостей с {source}...")
        headlines = fetch_news(source)

        for headline in headlines:
            sentiment = analyze_sentiment(headline)
            mentioned_cryptos = extract_crypto_mentions(headline)

            for crypto in mentioned_cryptos:
                crypto_sentiment[crypto].append(sentiment)

    # Анализ результатов
    results = []
    for crypto, sentiments in crypto_sentiment.items():
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            results.append({
                'crypto': crypto.capitalize(),
                'average_sentiment': avg_sentiment,
                'mentions': len(sentiments)
            })

    # Сортировка по среднему настроению и количеству упоминаний
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['average_sentiment', 'mentions'], ascending=[False, False])

    # Вывод всех баллов криптоактивов
    print("\nРезультаты анализа:")
    print(results_df.to_string(index=False))

    # Вывод рекомендации
    if not results_df.empty:
        best_crypto = results_df.iloc[0]
        if best_crypto['average_sentiment'] > 0 and best_crypto[
            'mentions'] >= 5:  # Фильтр по минимальному количеству упоминаний
            print("\nРекомендация:")
            print(f"Купить: {best_crypto['crypto']}")
            print(
                f"Причина: Положительное настроение в новостях (средний балл: {best_crypto['average_sentiment']:.2f}, упоминаний: {best_crypto['mentions']})")
        else:
            print(
                "\nРекомендация: Не покупать. Настроение в новостях нейтральное или негативное, либо недостаточно данных.")
    else:
        print("Не удалось найти достаточно данных для анализа.")


if __name__ == "__main__":
    main()