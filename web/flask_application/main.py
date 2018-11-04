from flask import Flask, render_template
from pymongo import MongoClient
import datetime
import pandas
import json

webApp = Flask(__name__)

# Date	Open	High	Low	Close	Volume	Adj Close
# 2016-07-01	17924.240234	18002.380859	17916.910156	17949.369141	82160000	17949.369141
# 2016-06-30	17712.759766	17930.609375	17711.800781	17929.990234	133030000	17929.990234
# 2016-06-29	17456.019531	17704.509766	17456.019531	17694.679688	106380000	17694.679688
# 2016-06-28	17190.509766	17409.720703	17190.509766	17409.720703	112190000	17409.720703
# 2016-06-27	17355.210938	17355.210938	17063.080078	17140.240234	138740000	17140.240234
# 2016-06-24	17946.630859	17946.630859	17356.339844	17400.75	239000000	17400.75
# 2016-06-23	17844.109375	18011.070312	17844.109375	18011.070312	98070000	18011.070312
# 2016-06-22	17832.669922	17920.160156	17770.359375	17780.830078	89440000	17780.830078
# 2016-06-21	17827.330078	17877.839844	17799.800781	17829.730469	85130000	17829.730469


@webApp.route('/setup')
def setup():
    stock_file = pandas.read_csv("DJIA_table.csv")
    article_file = pandas.read_csv("RedditNews.csv")

    stock_json = json.loads(stock_file.to_json(orient="records"))
    article_json = json.loads(article_file.to_json(orient="records"))

    client = MongoClient()
    db = client.fpa
    delete = db.stocks.delete_many({})
    delete = db.articles.delete_many({})

    db.articles.insert(article_json)
    db.stocks.insert(stock_json)
    # return render_template("homepage.html")
    return render_template("setup.html")

@webApp.route('/home')
def homepage():
    client = MongoClient()
    db = client.fpa

    list_articles = [article for article in db.articles.find({})]
    article_dates = [datetime.datetime.strptime(item['Date'], '%Y-%m-%d') for item in list_articles]
    article_titles = [item['News'] for item in list_articles]

    cut_article_dates = article_dates[:9]
    cut_article_titles = article_titles[:9]
    return render_template("homepage.html", Headlines = cut_article_titles)

@webApp.route('/stocks')
def stocks():
    client = MongoClient()
    db = client.fpa
    list_stocks = [stock for stock in db.stocks.find({})]
    dates = [datetime.datetime.strptime(item['date'], '%Y-%m-%d') for item in list_stocks]
    prices = [item['adj_close'] for item in list_stocks]
    cut_dates = dates[:10]
    cut_prices = prices[:10]
    most_recent_price = round(cut_prices[0], 2)
    for date in cut_dates:
        print(date)
    for price in cut_prices:
        print(price)
    return render_template("stocks.html", cut_prices=cut_prices, most_recent_price = most_recent_price, cut_dates = cut_dates)

@webApp.route('/feeds')
def feeds():
    return render_template("feeds.html")


if __name__ == "__main__":
    webApp.run(debug=True)