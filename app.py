import pandas as pd
import numpy as np
import pandas_datareader.data as web
from fbprophet import Prophet
import datetime
from flask import Flask, render_template
from flask import request, redirect
from pathlib import Path
import os
import os.path
import csv
from itertools import zip_longest

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
    
@app.route("/")
def first_page():
    tmp = Path("static/prophet.png")
    tmp_csv = Path("static/numbers.csv")
    if tmp.is_file():
        os.remove(tmp)
    if tmp_csv.is_file():
        os.remove(tmp_csv)
    return render_template("index.html")

#function to get crypto data
def yahoo_stocks(symbol, start, end):
    return web.DataReader(symbol, 'yahoo', start, end)

def get_historical_stock_price(stock):
    print ("Getting historical stock prices for stock ", stock)
    
    #get 7 year Crypto data  
    startDate = datetime.datetime(2022, 1, 1)
    #date = datetime.datetime.now().date()
    #endDate = pd.to_datetime(date)
    endDate = datetime.datetime(2022,12 , 1)
    stockData = yahoo_stocks(stock, startDate, endDate)
    return stockData

@app.route("/plot" , methods = ['POST', 'GET'] )
def main():
    if request.method == 'POST':
        stock = request.form['companyname']
        df_whole = get_historical_stock_price(stock)

        df = df_whole.filter(['Close'])
        
        df['ds'] = df.index
        #log transform the ‘Close’ variable to convert non-stationary data to stationary.
        df['y'] = np.log(df['Close'])
        original_end = df['Close'][-1]
        
        model = Prophet()
        model.fit(df)

        
        #num_days = int(input("Enter no of days to predict crypto price for: "))
        
        num_days = 5
        future = model.make_future_dataframe(periods=num_days)
        forecast = model.predict(future)
        
        print (forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        df.set_index('ds', inplace=True)
        forecast.set_index('ds', inplace=True)
        
        viz_df = df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
        viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])
        close_data = viz_df.Close
        forecasted_data = viz_df.yhat_scaled
        date = future['ds']
        forecast_start = forecasted_data[-num_days]

        d = [date, close_data, forecasted_data]
        export_data = zip_longest(*d, fillvalue = '')
        with open('static/numbers.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("Date", "Actual", "Forecasted"))
            wr.writerows(export_data)
        myfile.close()

        return render_template("plot.html", original = round(original_end,2), forecast = round(forecast_start,2), stock_tinker = stock.upper())

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
