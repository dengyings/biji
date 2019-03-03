import os
import sys
import datetime as dt
import numpy as np
import tushare as ts
import sklearn.covariance as cv
import sklearn.cluster as sc


def read_data(start_date, end_date):
    days = (end_date - start_date).days
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')

    stock = ts.get_sz50s()
    codes = stock.code.values
    names = stock.name.values

    opening_prices, closing_prices = [], []

    for code in codes:
        hist = ts.get_hist_data(code, start, end)
        dates = hist.index
        dates = np.array([dt.datetime.strptime(
            date, '%Y-%m-%d').date() for date in dates])
        dates = np.array(
            [delta_date.days for delta_date in dates - start_date])
        opens = hist.open.values
        closes = hist.close.values

        opening_price, closing_price = np.zeros(days + 1), np.zeros(days + 1)
        if len(dates) != 0:
            opening_price[dates] = opens
            closing_price[dates] = closes

        print('Opening price of stock {} from {} to {}:'.format(
            code, start, end), opening_price, sep='\n')
        print('Closing price of stock {} from {} to {}:'.format(
            code, start, end), closing_price, sep='\n')

        opening_prices.append(opening_price)
        closing_prices.append(closing_price)

    opening_prices = np.array(opening_prices)
    closing_prices = np.array(closing_prices)

    x = closing_prices - opening_prices
    no_zero_row = [x[i].any() != 0 for i in range(len(x))]
    x = x.compress(no_zero_row, axis=0)
    names = names.compress(no_zero_row)
    no_zero_std = [x[i].std() != 0 for i in range(len(x))]
    x = x.compress(no_zero_std, axis=0)
    names = names.compress(no_zero_std)
    x = (x / x.std(axis=1).reshape(-1, 1)).T

    return names, x


def train_model(x):
    model = cv.GraphLassoCV()

    with np.errstate(invalid='ignore'):
        try:
            model.fit(x)
        except FloatingPointError as error:
            print('FloatingPointError:', error)
            sys.exit(-1)

    return model


def pred_model(model, x):
    _, y = sc.affinity_propagation(model.covariance_)
    return y


def main(argc, argv, envp):
    start_date = dt.date(2017, 3, 9)
    end_date = dt.date(2017, 5, 9)

    names, x = read_data(start_date, end_date)
    model = train_model(x)
    pred_y = pred_model(model, x)

    for i in range(pred_y.max()):
        print('Class-{}: {}'.format(i, ', '.join(names[pred_y == i])))

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
