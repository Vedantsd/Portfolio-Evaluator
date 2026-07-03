from flask import Flask, render_template, request
import train as tr
import csv

app = Flask(__name__)
stock_data = {}

with open("stock_data.csv", 'r') as file: 
    reader = csv.DictReader(file)
    for row in reader : 
        stock_data[row['STOCK NAME']] = [row['SYMBOL'], row['LISTING DATE']]



@app.route("/")
def index() : 
    return render_template("index.html", data=stock_data)

if __name__ == "__main__" : 
    app.run(debug=True)