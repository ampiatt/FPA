from flask import Flask, render_template
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
    article_json = json.loads(article_file.to_json(orient="records"))

    db.stocks.delete_many({})
    db.articles.delete_many({})

    db.articles.insert(article_json)
    db.stocks.insert(stock_json)
    return render_template("setup.html")

@webApp.route('/home')
def homepage():
    client = MongoClient()
    db = client.fpa

    list_stocks = [stock for stock in db.stocks.find({})]
    dates = [datetime.datetime.strptime(item['date'], '%Y-%m-%d') for item in list_stocks]
    prices = [item['adj_close'] for item in list_stocks]
    list_articles = [article for article in db.articles.find({})]
    stock_prices = [item['adj_close'] for item in list_stocks]

    article_dates = [datetime.datetime.strptime(item['Date'], '%Y-%m-%d') for item in list_articles]
    article_titles = [item['News'] for item in list_articles]

    cut_stock_prices = stock_prices[:6]
    cut_article_titles = article_titles[:6]


    return render_template("homepage.html", Headlines = cut_article_titles, cut_prices = cut_stock_prices)

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

    return render_template("stocks.html", cut_prices=cut_prices, most_recent_price = most_recent_price, cut_dates = cut_dates)

@webApp.route('/feeds')
def feeds():
    return render_template("feeds.html")


if __name__ == "__main__":
    webApp.run(debug=True)