import train as tr
import csv

stock_data = {}

with open("stock_data.csv", 'r') as file: 
    reader = csv.DictReader(file)
    for row in reader : 
        stock_data[row['STOCK NAME']] = [row['SYMBOL'], row['LISTING DATE']]


stock_list = {}
k = int(input("Enter the number of stocks you want to predict: "))
for _ in range(k) : 
    sname = input("Enter a stock name: ").upper()

    if sname in stock_data.keys() : 
        stock_list[sname] = stock_data[sname]
    else : 
        print("Stock not found")


predicted_values = tr.predict_stock(stock_list)

for n, v in predicted_values.items() : 
    print(f"Stock name : {n}")
    print(f"Buy probability : {v[0]:.2%}")
    print(f"Signal : {v[1]:.3f}")
