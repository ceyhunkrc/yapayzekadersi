from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Selam! Flask uygulaması çalışıyor."

if __name__ == '__main__':
    app.run(debug=True)
