import time
import threading
import pandas as pd
import pandas_ta as ta
import logging
from pybit.unified_trading import HTTP
from key import API_KEY, API_SECRET
# Настройка логирования с повышенной точностью для маленьких значений
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Конфигурация API и торговых параметров

# Список символов для отслеживания
SYMBOLS = ["OBTUSDT", "PLUMEUSDT"]
INTERVAL = "1"  # таймфрейм – 1 минута
INITIAL_BALANCE = 500.0

# Инициализация клиента Bybit
client = HTTP(
    testnet=False,
    api_key=API_KEY,
    api_secret=API_SECRET
)

class TradingBot:
    def __init__(self, symbol):
        self.symbol = symbol
        self.demo_balance = INITIAL_BALANCE
        self.position = None

    def get_candle_data(self, limit: int = 20) -> pd.DataFrame:
        """
        Получение исторических данных за последние ~20 минут.
        При таймфрейме 1 минута будет получено 20 свечей.
        """
        try:
            response = client.get_kline(
                category="spot",
                symbol=self.symbol,
                interval=INTERVAL,
                limit=limit
            )
            if response['retCode'] != 0:
                logging.error("[%s] Ошибка API: %s", self.symbol, response['retMsg'])
                return pd.DataFrame()
            df = pd.DataFrame(response['result']['list'])
            df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            # Время приходит в миллисекундах
            df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
            df.set_index('open_time', inplace=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            # Переставляем строки, чтобы самая свежая свеча была в конце
            return df.iloc[::-1].copy()
        except Exception as e:
            logging.error("[%s] Ошибка получения данных: %s", self.symbol, e)
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет технических индикаторов для локального анализа:
          - EMA5 и EMA10 (короткие скользящие средние)
          - RSI с периодом 7
          - ROC (процентное изменение цены за 5 свечей)
        """
        try:
            df['ema5'] = ta.ema(df['close'], length=5)
            df['ema10'] = ta.ema(df['close'], length=10)
            df['rsi'] = ta.rsi(df['close'], length=7)
            df['roc'] = ta.roc(df['close'], length=5)
            return df.dropna()
        except Exception as e:
            logging.error("[%s] Ошибка расчета индикаторов: %s", self.symbol, e)
            return pd.DataFrame()

    def analyze_signals(self, df: pd.DataFrame) -> dict:
        """
        Анализ торговых сигналов на основе следующих условий:
          - Сигнал на покупку:
              • EMA5 пересекает EMA10 снизу вверх
              • RSI ниже 40 (локальная перепроданность)
              • ROC > 1 (рост цены более чем на 1% за 5 свечей)
          - Сигнал на продажу:
              • EMA5 пересекает EMA10 сверху вниз
              • RSI выше 60 (локальная перекупленность)
              • ROC < -1 (падение цены более чем на 1%)
        """
        if len(df) < 2:
            return {'buy': False, 'sell': False}
        prev = df.iloc[-2]
        last = df.iloc[-1]
        buy_signal = (
            (prev['ema5'] < prev['ema10']) and
            (last['ema5'] > last['ema10']) and
            (last['rsi'] < 40) and
            (last['roc'] > 1)
        )
        sell_signal = (
            (prev['ema5'] > prev['ema10']) and
            (last['ema5'] < last['ema10']) and
            (last['rsi'] > 60) and
            (last['roc'] < -1)
        )
        return {'buy': buy_signal, 'sell': sell_signal}

    def execute_buy(self, price: float):
        """Симуляция покупки актива"""
        if self.demo_balance <= 0:
            return
        self.position = {
            'entry_price': price,
            'quantity': self.demo_balance / price,
            'timestamp': pd.Timestamp.now()
        }
        invested = self.position['quantity'] * price
        self.demo_balance = 0.0
        logging.info("[%s] BUY ORDER | Price: %.6f | Qty: %.6f | Invested: %.2f USD",
                     self.symbol, price, self.position['quantity'], invested)

    def execute_sell(self, price: float):
        """Симуляция продажи актива"""
        if not self.position:
            return
        profit = (price - self.position['entry_price']) * self.position['quantity']
        self.demo_balance = self.position['quantity'] * price
        logging.info("[%s] SELL ORDER | Price: %.6f | Profit: %.2f USD | New Balance: %.2f USD",
                     self.symbol, price, profit, self.demo_balance)
        self.position = None

    def log_status(self, df: pd.DataFrame):
        """Логирование текущего состояния (цена и индикаторы) с указанием символа"""
        last = df.iloc[-1]
        logging.info("[%s] STATUS | Price: %.6f | EMA5/10: %.6f/%.6f | RSI: %.2f | ROC: %.6f",
                     self.symbol, last['close'], last['ema5'], last['ema10'], last['rsi'], last['roc'])

    def run(self):
        """
        Основной цикл работы торгового бота для данной монеты.
        Данные обновляются каждую минуту (новая свеча формируется каждые 60 секунд).
        Анализ ведется на окне из 20 свечей (~20 минут).
        """
        logging.info("[%s] Starting trading bot with initial balance: %.2f USD", self.symbol, INITIAL_BALANCE)
        while True:
            try:
                raw_data = self.get_candle_data(limit=20)
                if raw_data.empty:
                    time.sleep(60)
                    continue
                df = self.calculate_indicators(raw_data)
                if df.empty:
                    time.sleep(60)
                    continue
                signals = self.analyze_signals(df)
                current_price = df.iloc[-1]['close']
                self.log_status(df)
                if self.position:
                    if signals['sell']:
                        self.execute_sell(current_price)
                else:
                    if signals['buy']:
                        self.execute_buy(current_price)
                time.sleep(60)
            except KeyboardInterrupt:
                logging.info("[%s] Bot stopped by user", self.symbol)
                break
            except Exception as e:
                logging.error("[%s] Critical error: %s", self.symbol, e)
                time.sleep(60)

if __name__ == "__main__":
    # Создаем экземпляры ботов для каждой монеты из списка
    bots = [TradingBot(symbol) for symbol in SYMBOLS]
    threads = []
    for bot in bots:
        t = threading.Thread(target=bot.run, daemon=True)
        threads.append(t)
        t.start()
    # Основной поток остается активным, чтобы не завершились дочерние потоки
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Остановка всех ботов по требованию пользователя")
