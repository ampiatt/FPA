from flask import Flask, render_template
from pymongo import MongoClient

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


@webApp.route('/')
def setup():
    client = MongoClient()
    db = client.fpa
    db.stocks.insert_one({'date': '2016-06-21', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-23', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-22', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-25', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-26', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-27', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-28', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-29', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-25', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-26', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    db.stocks.insert_one({'date': '2016-06-27', 'open': '17827.330078', 'high': '17877.839844', 'low': '17799.800781',
                          'close': '17829.730469', 'volume': '85130000', 'adj_close': '17829.730469'})
    return render_template("homepage.html")

@webApp.route('/home')
def homepage():
    return render_template("homepage.html")

@webApp.route('/stocks')
def stocks():
    client = MongoClient()
    db = client.fpa
    list_stocks = db.stocks.find({})
    print(type(list_stocks))
    for stock in db.stocks.find({}):
        print(stock)
    return render_template("stocks.html", stocks=list_stocks)

@webApp.route('/feeds')
def feeds():
    return render_template("feeds.html")


if __name__ == "__main__":
    webApp.run(debug=True)