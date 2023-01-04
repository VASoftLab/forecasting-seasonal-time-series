import os
import time
import datetime as dt
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet

from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import SARIMAX

from collections import namedtuple
from tabulate import tabulate

import pmdarima as pm

# Глобальная переменная для хранения настроек фигур
FigureSettings = namedtuple('FigureSettings', ['figWidth', 'figHeight', 'dpi', 'labelXSize', 'labelYSize',
                                               'tickMajorLabelSize', 'tickMinorLabelSize', 'tickXLabelRotation',
                                               'markerSize', 'legendFontSize', 'titleFontSize', 'boxTextSize'])
fset = FigureSettings
fset.figWidth = 10
fset.figHeight = 6
fset.dpi = 300
fset.labelXSize = 14
fset.labelYSize = 14
fset.tickMajorLabelSize = 12
fset.tickMinorLabelSize = 10
fset.tickXLabelRotation = 30
fset.markerSize = 5
fset.legendFontSize = 14
fset.titleFontSize = 14
fset.boxTextSize = 14

# Путь к корневой папке проекта
dir_name = os.path.dirname(__file__)


# Предварительная обработка данных
def data_preprocessing(report_file_name: str, column_name: str, data_file_name: str):
    """
    Функция предварительной обработки данных
    :param report_file_name: string - Полный путь к входному файлу с данными
    :param column_name: string - Имя колонки, которая будет использована как источник данных
    :param data_file_name: string - Полный путь к выходному файлу с данными
    """
    start_time = time.time()

    # Чтение данных
    print(f'Reading data from {report_file_name}')
    df = pd.read_csv(report_file_name, parse_dates=['DT'])
    print('\nInitial dataframe')
    print(df.head())

    # Новый датафрейм
    new_df = df[['DT', column_name]]
    # Установка индекса
    new_df = new_df.set_index('DT')
    new_df = new_df[(new_df.index >= dt.datetime(2022, 12, 12)) &
                    (new_df.index < dt.datetime(2022, 12, 31))]
    # Передискретизация / Resample
    # W - неделя
    # D - календарный день
    # H - час
    # T - минута
    # S - секунда
    avg_df = pd.DataFrame(data=new_df[column_name].resample('H').mean())
    # Изменяем заголовок столбца
    avg_df.columns = ['Measurement']
    print('\nResampled dataframe')
    print(avg_df.head())
    # Сохранить новый датафрейм в файл
    avg_df.to_csv(data_file_name)
    print(f'\nNew file saved to {data_file_name}')
    print(f'\nExecution time: {(time.time() - start_time):3.2f} sec.')

    # Визуализация
    values = avg_df['Measurement']
    fig, ax = plt.subplots(figsize=(fset.figWidth, fset.figHeight))
    fig.set_dpi(fset.dpi)
    ax.plot(values)
    ax.axvline(x=dt.datetime(2022, 12, 29), color='g', alpha=0.5)
    ax.axvline(x=dt.datetime(2022, 12, 31), color='g', alpha=0.5)

    ax.axvspan(dt.datetime(2022, 12, 29),
               dt.datetime(2022, 12, 31), color='g', alpha=0.2)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(dt.datetime(2022, 12, 21), 8.5, "   Train Data   ",
            ha="center", va="center", size=fset.boxTextSize, bbox=bbox_props)
    ax.text(dt.datetime(2022, 12, 30), 8.5, "Test Data",
            ha="center", va="center", size=fset.boxTextSize, bbox=bbox_props)

    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.title('History Horizont', style='italic', fontsize=fset.labelXSize)
    # plt.xlabel('History Horizont', style='italic', fontsize=fset.labelXSize)
    plt.ylabel('Measurement', style='italic', fontsize=fset.labelYSize)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis='both', which='major', labelsize=fset.tickMajorLabelSize)
    ax.tick_params(axis='both', which='minor', labelsize=fset.tickMinorLabelSize)

    plt.setp(ax.get_xticklabels(), rotation=fset.tickXLabelRotation)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'img', 'Fig1.png')), dpi=fset.dpi,
                bbox_inches='tight')

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


# Оценка производительности метода
def performance_evaluation(y_true: pd.DataFrame, y_pred: pd.DataFrame, method_name: str,
                           save_to_file: bool = True, overwrite: bool = False):

    r2_value = r2(y_true, y_pred)
    mse_value = mse(y_true, y_pred)
    rmse_value = mse(y_true, y_pred, squared=False)
    mae_value = mae(y_true, y_pred)
    mape_value = mape(y_true, y_pred)

    print(f'\nPerformance Evaluation for {method_name}')
    print(f'R2: {r2_value:3.2f}')
    print(f'MSE: {mse_value:3.2f}')
    print(f'RMSE: {rmse_value:3.2f}')
    print(f'MAE: {mae_value:3.2f}')
    print(f'MAPE: {mape_value:3.2f}%')

    if save_to_file:
        report_file = os.path.realpath(os.path.join(dir_name, '..', 'data', 'forecasting-report.txt'))

        if overwrite:
            f = open(report_file, 'w')
        else:
            f = open(report_file, 'a')

        print(f'#################### {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} ####################', file=f)
        print(f'Performance Evaluation for {method_name.upper()} method\n', file=f)

        table = [['METRIC', 'VALUE'],
                 ['R2', f'{r2_value:3.2f}'],
                 ['MSE', f'{mse_value:3.2f}'],
                 ['RMSE', f'{rmse_value:3.2f}'],
                 ['MAE', f'{mae_value:3.2f}'],
                 ['MAPE', f'{mape_value:3.2f}%']]
        print(tabulate(table, headers='firstrow', tablefmt='github'), file=f)
        print('\n', file=f)

        f.close()


def performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper, method_name, fig_file_name):
    fig, ax = plt.subplots(figsize=(fset.figWidth, fset.figHeight))
    fig.set_dpi(fset.dpi)
    ax.plot(x_date, y_pred, '-')
    ax.plot(x_date, y_true, 'o', color='tab:brown', markersize=fset.markerSize)
    ax.fill_between(x_date, ypred_lower, ypred_upper, alpha=0.2)
    ax.legend(['Predicted Value', 'Actual Value'], fontsize=fset.legendFontSize, loc='lower right')

    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.title(f'{method_name.upper()} Forecast Horizont', style='italic', fontsize=fset.titleFontSize)
    plt.ylabel('Measurement', style='italic', fontsize=fset.labelYSize)

    plt.ylim(2, 9)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis='both', which='major', labelsize=fset.tickMajorLabelSize)
    ax.tick_params(axis='both', which='minor', labelsize=fset.tickMinorLabelSize)
    plt.setp(ax.get_xticklabels(), rotation=fset.tickXLabelRotation)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'img', fig_file_name)),
                dpi=fset.dpi, bbox_inches='tight')

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


# Прогнозирование с помощью метода Prophet
def prophet_forecasting(data_file_name: str):
    # https://facebook.github.io/prophet/docs/quick_start.html
    start_time = time.time()
    df = pd.read_csv(data_file_name, parse_dates=['DT'])
    df.rename(columns={'DT': 'ds', 'Measurement': 'y'}, inplace=True)
    print('\nProphet dataframe YYYY-MM-DD HH:MM:SS')
    print(df)

    # Обучающая и тестовая выборки
    df_train = df[(df['ds'] >= dt.datetime(2022, 12, 12)) & (df['ds'] < dt.datetime(2022, 12, 29))]
    print('\nTrain dataset')
    print(df_train)

    df_test = df[(df['ds'] >= dt.datetime(2022, 12, 29)) & (df['ds'] < dt.datetime(2022, 12, 31))]
    print('\nTest dataset')
    print(df_test)

    # Создание объекта Prophet
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=48, freq='60min', include_history=False)
    print('\nProphet future')
    print(future)

    # Прогноз
    forecast = m.predict(future)
    print('\nforecast')
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    print(f'\nExecution time: {(time.time() - start_time):3.2f} sec.')

    x_date = forecast['ds']
    y_true = df_test['y']
    y_pred = forecast['yhat']
    ypred_lower = forecast['yhat_lower']
    ypred_upper = forecast['yhat_upper']

    performance_evaluation(y_true, y_pred, 'Prophet', overwrite=True)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper, 'Prophet', 'Fig2.png')


# Настройка модели SARIMA
def sarima_tuning(data_file_name: str):
    start_time = time.time()
    report_file = os.path.realpath(os.path.join(dir_name, '..', 'data', 'forecasting-report.txt'))
    f = open(report_file, 'a')

    df = pd.read_csv(data_file_name, parse_dates=['DT'])
    df.rename(columns={'DT': 'ds', 'Measurement': 'y'}, inplace=True)

    # Обучающая и тестовая выборки
    df_train = df[(df['ds'] >= dt.datetime(2022, 12, 12)) & (df['ds'] < dt.datetime(2022, 12, 29))]
    df_test = df[(df['ds'] >= dt.datetime(2022, 12, 29)) & (df['ds'] < dt.datetime(2022, 12, 31))]

    # Данные
    data = df_train['y']

    # Augmented Dickey-Fuller test
    adf = adfuller(data, autolag='AIC')
    print(f'Augmented Dickey-Fuller Test for SARIMA method\n')
    print(f'#################### {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} ####################', file=f)
    print(f'Augmented Dickey-Fuller Test for SARIMA method\n', file=f)
    print("1. ADF : ", adf[0])
    print("1. ADF : ", adf[0], file=f)
    print("2. P-Value : ", adf[1])
    print("2. P-Value : ", adf[1], file=f)
    print("3. Num Of Lags : ", adf[2])
    print("3. Num Of Lags : ", adf[2], file=f)
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", adf[3])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", adf[3], file=f)
    print("5. Critical Values :")
    print("5. Critical Values :", file=f)
    for key, val in adf[4].items():
        print("\t", key, ": ", val)
        print("\t", key, ": ", val, file=f)

    # Подбор параметров
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html#pmdarima.arima.auto_arima
    print('\nAutoARIMA\n')
    print(f'#################### {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} ####################', file=f)
    print(f'AutoARIMA model summary\n', file=f)
    m = pm.arima.auto_arima(data,
                            start_p=1,
                            start_q=1,
                            max_p=3,
                            max_q=3,
                            m=24,
                            start_P=0,
                            seasonal=True,
                            d=None,
                            D=1,
                            test='adf',
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

    print(m.summary())
    print(m.summary(), file=f)
    f.close()

    print(f'\nSARIMA tuning execution time: {(time.time() - start_time):3.2f} sec.')


# Прогнозирование с помощью метода SARIMA
def sarima_forecasting(data_file_name: str):
    start_time = time.time()
    report_file = os.path.realpath(os.path.join(dir_name, '..', 'data', 'forecasting-report.txt'))
    f = open(report_file, 'a')

    df = pd.read_csv(data_file_name, parse_dates=['DT'])
    df.rename(columns={'DT': 'ds', 'Measurement': 'y'}, inplace=True)

    # Обучающая и тестовая выборки
    df_train = df[(df['ds'] >= dt.datetime(2022, 12, 12)) & (df['ds'] < dt.datetime(2022, 12, 29))]
    df_test = df[(df['ds'] >= dt.datetime(2022, 12, 29)) & (df['ds'] < dt.datetime(2022, 12, 31))]

    # Данные
    data = df_train['y']
    m = SARIMAX(data, order=(3, 0, 2), seasonal_order=(0, 1, 2, 24),
                enforce_stationarity=False, enforce_invertibility=False).fit()
    print(m.summary())
    print(f'#################### {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} ####################', file=f)
    print(f'SARIMA model summary\n', file=f)
    print(m.summary(), file=f)

    forecast = m.get_forecast(steps=48, signal_only=True)
    forecast_interval = forecast.conf_int()

    print('\nSARIMA forecasting:')
    print(forecast.predicted_mean)
    print('\nSARIMA forecasting interval:')
    print(forecast_interval)

    data_range = pd.date_range('2022-12-29', '2022-12-31', freq='H')
    data_frame = data_range.to_frame().iloc[:-1, :]
    data_frame.columns = ['ds']

    print(f'\nSARIMA execution time: {(time.time() - start_time):3.2f} sec.')

    x_date = data_frame['ds']
    y_true = df_test['y']
    y_pred = forecast.predicted_mean
    ypred_lower = forecast_interval.iloc[:, 0]
    ypred_upper = forecast_interval.iloc[:, 1]

    performance_evaluation(y_true, y_pred, 'SARIMA')
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper, 'SARIMA', 'Fig3.png')


if __name__ == '__main__':
    # File names
    rep_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'report_2022_12_8__2023_1_1.csv'))
    dat_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'data.csv'))

    # Data column name
    col_name = 'Influent'

    # Data Preprocessing
    # data_preprocessing(rep_file_name, col_name, dat_file_name)

    # Prophet Forecasting
    prophet_forecasting(dat_file_name)

    # SARIMA Tuning
    # sarima_tuning(dat_file_name)
    # SARIMA Forecasting
    sarima_forecasting(dat_file_name)
