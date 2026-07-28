import pandas as pd
import yfinance as yf
import datetime
from xgboost import XGBClassifier

d = datetime.datetime.now()
end = d.date()


def predict_stock(stock_data):

    predicted_values = {}
    errors = {}

    for sname in stock_data.keys():

        try:
            stock_detais = stock_data[sname]
            stock_name = stock_detais[0]
            start = stock_detais[1]

            data = yf.download(stock_name, start=start, end=end, progress=False)

            if data is None or data.empty:
                errors[sname] = f"No data returned for symbol '{stock_name}'. Check the symbol and listing date."
                continue

            df = pd.DataFrame(data)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if not required_cols.issubset(set(df.columns)):
                errors[sname] = f"Downloaded data for '{stock_name}' is missing expected columns."
                continue

            df['MVA10'] = df['Close'].rolling(window=10).mean()
            df['MVA20'] = df['Close'].rolling(window=20).mean()
            df['MVA50'] = df['Close'].rolling(window=50).mean()

            df['Return'] = df['Close'].pct_change(fill_method=None)

            df['Volatility'] = df['Return'].rolling(window=10).std()

            delta = df['Close'].diff()

            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()

            df['MACD'] = ema12 - ema26

            df['Volume_MA20'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']

            df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
            df['Target'] = (df['Future_Return'] > 0.02).astype(int)

            df = df.dropna()


            MIN_ROWS = 60
            if len(df) < MIN_ROWS:
                errors[sname] = (
                    f"Not enough trading history for '{stock_name}' after feature "
                    f"calculation ({len(df)} usable rows, need at least {MIN_ROWS}). "
                    f"Try a stock with a longer/older listing date."
                )
                continue

            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MVA10', 'MVA20',
                        'MVA50', 'Return', 'Volatility', 'RSI', 'MACD', 'Volume_Ratio']

            X = df[features]
            y = df['Target']

            if y.nunique() < 2:
                errors[sname] = (
                    f"'{stock_name}' has only one outcome class in the training "
                    f"window (no up or no down days beyond the 2% threshold), "
                    f"so a model can't be trained."
                )
                continue

            split = int(len(df) * 0.8)

            X_train = X[:split]
            X_test = X[split:]

            y_train = y[:split]
            y_test = y[split:]

            if len(X_train) == 0 or y_train.nunique() < 2:
                errors[sname] = f"Not enough varied training data for '{stock_name}'."
                continue

            model = XGBClassifier(n_estimators=300, learning_rate=0.5, max_depth=4,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train, y_train)

            latest = X.iloc[[-1]]

            buy_probability = model.predict_proba(latest)[0, 1]
            signal = 2 * buy_probability - 1

            predicted_values[sname] = [buy_probability, signal]

        except Exception as e:
            errors[sname] = f"Unexpected error: {e}"
            continue

    return predicted_values, errors