import numpy as np
import pandas as pd
import mwclient
import time
from datetime import datetime
from transformers import pipeline
import yfinance as yf
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier


site = mwclient.Site("en.wikipedia.org")
page = site.pages["Bitcoin"]
revs = list(page.revisions())[:1000]
revs = sorted(revs, key = lambda rev: rev["timestamp"])
sentiment_pipeline = pipeline("sentiment-analysis")

# Sentiment Analysis
def find_sentiment(text):
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "Negative":
        score *= -1
    return score
edits = {}

# Sentiment in Numbers
for rev in revs:
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    if date not in edits:
        edits[date] = dict(sentiments = list(), edit_count=0)
    
    edits[date]["edit_count"] += 1
    edits[date]["sentiments"].append(find_sentiment(rev.get('comment', '')))
    from statistics import mean

# Sentiment Data Cleaning
for key in edits:
    if len(edits[key]["sentiments"]) > 0:
        edits[key]["sentiment"] = mean(edits[key]["sentiments"])
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s > 0]) / len(edits[key]["sentiments"])
    else:
        edits[key]["sentiment"] = 0
        edits[key]["neg_sentiment"] = 0
    del edits[key]["sentiments"]
    
    
edits_df = pd.DataFrame.from_dict(edits, orient='index')
dates = pd.date_range(start = '2020-03-08', end = datetime.today())
rolling_edits = edits_df.rolling(30).mean()
rolling_edits = rolling_edits.dropna()
rolling_edits.to_csv("D:\Anaconda\app\bitcoin-predict\wikipedia_edits.csv")

btc_ticker = yf.Ticker('BTC-USD') 
btc = btc_ticker.history(period='max')

btc.index = pd.to_datetime(btc.index)
btc.index = btc.index.tz_localize(None)
del btc['Dividends']
del btc['Stock Splits']
btc.columns = [i.lower() for i in btc.columns]

btc.plot.line(y = 'close', use_index = True)
wiki = pd.read_csv('wikipedia_edits.csv', index_col = 0, parse_dates = True)
btc = btc.merge(wiki, left_index=True, right_index=True)
btc['tomorrow'] = btc['close'].shift(-1)
btc['target'] = (btc['tomorrow'] > btc['close']).astype(int)
btc['target'].value_counts()



model = RandomForestClassifier(n_estimators = 10, min_samples_split=50, random_state=1)

train = btc.iloc[:-200]
test = btc[-200:]
predictors = ['close', 'volume', 'open', 'high', 'low', 'edit_count', 'sentiment', 'neg_sentiment']
model.fit(train[predictors], train['target'])


preds = model.predict(test[predictors])
preds = pd.Series(preds, index = test.index)
precision_score(test['target'], preds)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name='predictions')
    combined = pd.concat([test['target'], preds], axis=1)
    return combined

print("BTC DataFrame rows:", btc.shape[0])

def backtest(data, model, predictors, start=1095, step=150):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        print(f"Iteration with i={i}")  # Debug print
        train = data.iloc[0:i].copy()
        test = data.iloc[:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    print("Total iterations:", len(all_predictions))
    return pd.concat(all_predictions) if all_predictions else pd.DataFrame()


model = XGBClassifier(random_state = 1, learning_rate = .1, n_estimators = 50)
predictions = backtest(btc, model, predictors, start=365, step=150)
precision_score(predictions['target'], predictions['predictions'])

def compute_rolling(btc):
    horizons = [7, 60]
    new_predictors = ['close', 'sentiment', 'neg_sentiment']
    
    for horizon in horizons:
        rolling_averages = btc.rolling(horizon, min_periods=1).mean()
        
        ratio_column = f'close_ratio_{horizon}'
        btc[ratio_column] = btc['close'] / rolling_averages['close']
        
        edit_column = f'edit_{horizon}'
        btc[edit_column] = rolling_averages['edit_count']
        
        rolling = btc.rolling(horizon, closed='left', min_periods=1).mean()
        trend_column = f'trend_{horizon}'
        btc[trend_column] = rolling['target']
        new_predictors += [ratio_column, trend_column, edit_column]
    return btc, new_predictors


btc, new_predictors = compute_rolling(btc.copy())
predictions = backtest(btc, model, new_predictors)
precision_score(predictions['target'], predictions['predictions'])
print(predictions)


