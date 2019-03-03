import os
import sys
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm


def read_data():
    housing = sd.load_boston()
    x, y = su.shuffle(housing.data, housing.target,
                      random_state=7)
    return x, y


def train_model_dt(x, y):
    model_dt = st.DecisionTreeRegressor(max_depth=4)
    model_dt.fit(x, y)
    return model_dt


def train_model_ab(x, y):
    model_ab = se.AdaBoostRegressor(
        st.DecisionTreeRegressor(max_depth=4),
        n_estimators=400, random_state=7)
    model_ab.fit(x, y)
    return model_ab


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


def main(argc, argv, envp):
    x, y = read_data()
    train_size = int(len(x) * 0.8)
    train_x = x[:train_size]
    train_y = y[:train_size]
    model_dt = train_model_dt (train_x, train_y)
    model_ab = train_model_ab (train_x, train_y)
    test_x = x[train_size:]
    test_y = y[train_size:]
    pred_test_y_dt = pred_model(model_dt, test_x)
    pred_test_y_ab = pred_model(model_ab, test_x)
    eval_model(test_y, pred_test_y_dt)
    eval_model(test_y, pred_test_y_ab)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
