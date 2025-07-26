from flask import Flask

print("printing from test.py")

app = Flask(__name__)
@app.route('/')
def hello():
    return "Hello, World!"
