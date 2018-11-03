from flask import Flask, render_template

webApp = Flask(__name__)

@webApp.route('/')
def homepage():
    return render_template("homepage.html")


if __name__ == "__main__":
    webApp.run(debug=True)