from flask import Flask
import datetime

app = Flask(__name__)

@app.route('/test')
def home():
    return "Hello, World!"

def scheduled_task():
    print(f"Scheduled task running at {datetime.datetime.now()}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
