import os
import sys
import csv
import platform
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp


def read_data(filename, fb, fe):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x, y = [], []
        for row in reader:
            x.append(row[fb:fe])
            y.append(row[-1])
        fn = np.array(x[0])
        x = np.array(x[1:], dtype=float)
        y = np.array(y[1:], dtype=float)
        x, y = su.shuffle(x, y, random_state=7)
    return fn, x, y


def train_model(x, y):
    model = se.RandomForestRegressor(max_depth=10,
        n_estimators=1000, min_samples_split=2)
    model.fit(x, y)
    return model


def pred_model(model, x):
    y = model.predict(x)
    return y


def eval_model(y, pred_y):
    mae = sm.mean_absolute_error(y, pred_y)
    mse = sm.mean_squared_error(y, pred_y)
    mde = sm.median_absolute_error(y, pred_y)
    evs = sm.explained_variance_score(y, pred_y)
    r2s = sm.r2_score(y, pred_y)
    print(mae, mse, mde, evs, r2s)


def init_model_day():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(211)
    mp.title('Random Forest Regression By Day', fontsize=16)
    mp.xlabel('Feature', fontsize=12)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(axis='y', linestyle=':')


def draw_model_day(fn_day, fi_day):
    fi_day = (fi_day * 100) / fi_day.max()
    sorted_indices = np.flipud(fi_day.argsort())
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_day[sorted_indices], align='center',
           facecolor='deepskyblue', edgecolor='steelblue',
           label='Day')
    mp.xticks(pos, fn_day[sorted_indices])
    mp.legend()


def init_model_hour():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(212)
    mp.title('Random Forest Regression By Hour', fontsize=16)
    mp.xlabel('Feature', fontsize=12)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(axis='y', linestyle=':')


def draw_model_hour(fn_hour, fi_hour):
    fi_hour = (fi_hour * 100) / fi_hour.max()
    sorted_indices = np.flipud(fi_hour.argsort())
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_hour[sorted_indices], align='center',
           facecolor='lightcoral', edgecolor='indianred',
           label='Hour')
    mp.xticks(pos, fn_hour[sorted_indices])
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.tight_layout()
    mp.show()


def main(argc, argv, envp):
    fn_day, x_day, y_day = read_data('bike_day.csv', 2, 13)
    train_size_day = int(len(x_day) * 0.8)
    train_x_day = x_day[:train_size_day]
    train_y_day = y_day[:train_size_day]
    model_day = train_model (train_x_day, train_y_day)
    test_x_day = x_day[train_size_day:]
    test_y_day = y_day[train_size_day:]
    pred_test_y_day = pred_model(model_day, test_x_day)
    eval_model(test_y_day, pred_test_y_day)
    fi_day = model_day.feature_importances_
    fn_hour, x_hour, y_hour = read_data('bike_hour.csv', 2, 14)
    train_size_hour = int(len(x_hour) * 0.8)
    train_x_hour = x_hour[:train_size_hour]
    train_y_hour = y_hour[:train_size_hour]
    model_hour = train_model (train_x_hour, train_y_hour)
    test_x_hour = x_hour[train_size_hour:]
    test_y_hour = y_hour[train_size_hour:]
    pred_test_y_hour = pred_model(model_hour, test_x_hour)
    eval_model(test_y_hour, pred_test_y_hour)
    fi_hour = model_hour.feature_importances_
    init_model_day()
    draw_model_day(fn_day, fi_day)
    init_model_hour()
    draw_model_hour(fn_hour, fi_hour)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
