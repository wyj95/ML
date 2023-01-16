from flask import Flask

app = Flask(__name__)

@app.route("/")
def f():
    return "SB YC"

app.run(port=8083)
