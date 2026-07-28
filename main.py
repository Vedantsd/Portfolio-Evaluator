from flask import Flask, render_template, request
import train as tr
import csv

app = Flask(__name__)

stock_data = {}
with open("stock_data.csv", 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        stock_data[row['STOCK NAME']] = [row['SYMBOL'], row['LISTING DATE']]



@app.route("/")
def index():
    return render_template("index.html", data=stock_data)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    selected_names = request.form.getlist("stocks")

    if not selected_names:
        return render_template("results.html", results={}, errors={})

    selected_stock_data = {
        name: stock_data[name] for name in selected_names if name in stock_data
    }

    predicted_values, errors = tr.predict_stock(selected_stock_data)

    return render_template("results.html", results=predicted_values, errors=errors)


if __name__ == "__main__":
    app.run(debug=True)