from flask import Flask, render_template, stream_with_context
from werkzeug.datastructures import Headers
from werkzeug.wrappers import Response
from pymongo import MongoClient
import datetime
import pandas
import json

webApp = Flask(__name__)



@webApp.route('/setup')
def setup():
    stock_file = pandas.read_csv("DJIA_table.csv")
    article_file = pandas.read_csv("RedditNews.csv")

    client = MongoClient()
    db = client.fpa

    stock_json = json.loads(stock_file.to_json(orient="records"))
    for value in stock_json:
        value['date'] = int((datetime.datetime.strptime(value['date'], '%Y-%m-%d') - datetime.datetime(1970, 1, 1)).total_seconds())
    article_json = json.loads(article_file.to_json(orient="records"))
    print(article_json)
    for value in article_json:
        value['date'] = int((datetime.datetime.strptime(value['Date'], '%Y-%m-%d') - datetime.datetime(1970, 1, 1)).total_seconds())
        del value['Date']

    db.stocks.delete_many({})
    db.articles.delete_many({})

    db.articles.insert(article_json)
    db.stocks.insert(stock_json)
    return render_template("setup.html")

def get_data():
    client = MongoClient()
    db = client.fpa

    # read everything out of the db
    data = {}
    for stock in db.stocks.find({}):
        data[stock['date']] = {'actual': stock['close']}
    for prediction in db.predictions.find({}):
        if prediction['date'] not in data:
            continue
        data[prediction['date']]['prediction'] = prediction['price']
    keys = [key for key in data.keys()]
    for key in keys:
        if 'prediction' not in data[key]:
            del data[key]
    return data

@webApp.route('/home')
def homepage():
    data = get_data()
    # generate graph data
    dates = sorted(data.keys())[:-10]
    labels = [datetime.datetime.utcfromtimestamp(date).strftime('%Y-%m-%d') for date in dates]
    prices = [data[date]['actual'] for date in dates]
    predictions = [data[date]['prediction'] for date in dates]

    return render_template("homepage.html", prices=prices, predictions=predictions, labels=labels)

@webApp.route('/csv')
def csv_download():
    def generate():
        data = get_data()
        output = "date, actual, prediction\n"
        for date in data:
            output += "{},{},{}\n".format(datetime.datetime.utcfromtimestamp(date).strftime('%Y-%m-%d'), data[date]['actual'], data[date]['prediction'])
        return output
    headers = Headers()
    headers.set('Content-Disposition', 'attachment', filename='stock_price_predictions.csv')
    return Response(
        stream_with_context(generate()),
        mimetype='text/csv', headers=headers
    )

if __name__ == "__main__":
    webApp.run()
