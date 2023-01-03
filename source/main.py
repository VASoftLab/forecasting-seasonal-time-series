import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet

from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

from collections import namedtuple
from tabulate import tabulate

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
    new_df = new_df[(new_df.index >= datetime.datetime(2022, 12, 12)) &
                    (new_df.index < datetime.datetime(2022, 12, 31))]
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
    ax.axvline(x=datetime.datetime(2022, 12, 29), color='g', alpha=0.5)
    ax.axvline(x=datetime.datetime(2022, 12, 31), color='g', alpha=0.5)

    ax.axvspan(datetime.datetime(2022, 12, 29),
               datetime.datetime(2022, 12, 31), color='g', alpha=0.2)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(datetime.datetime(2022, 12, 21), 8.5, "   Train Data   ",
            ha="center", va="center", size=fset.boxTextSize, bbox=bbox_props)
    ax.text(datetime.datetime(2022, 12, 30), 8.5, "Test Data",
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


def write_report(r2_value, mse_value, rmse_value, mae_value, mape_value, method_name, overwrite=False):
    report_file = os.path.realpath(os.path.join(dir_name, '..', 'data', 'forecasting-report.txt'))

    if overwrite:
        f = open(report_file, 'w')
    else:
        f = open(report_file, 'a')

    print('##################################################', file=f)
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


def performance_evaluation(method_name, y_true, y_pred):
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

    return r2_value, mse_value, rmse_value, mae_value, mape_value


def prophet_evaluation(xdat, meas, yhat, yhat_lower, yhat_upper):
    fig, ax = plt.subplots(figsize=(fset.figWidth, fset.figHeight))
    fig.set_dpi(fset.dpi)
    ax.plot(xdat, yhat, '-')
    ax.plot(xdat, meas, 'o', color='tab:brown', markersize=fset.markerSize)
    ax.fill_between(xdat, yhat_lower, yhat_upper, alpha=0.2)
    ax.legend(['Predicted Value', 'Actual Value'], fontsize=fset.legendFontSize)

    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.title('Forecast Horizont', style='italic', fontsize=fset.labelXSize)
    # plt.xlabel('Forecast Horizont', style='italic', fontsize=fset.labelXSize)
    plt.ylabel('Measurement', style='italic', fontsize=fset.labelYSize)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis='both', which='major', labelsize=fset.tickMajorLabelSize)
    ax.tick_params(axis='both', which='minor', labelsize=fset.tickMinorLabelSize)
    plt.setp(ax.get_xticklabels(), rotation=fset.tickXLabelRotation)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'img', 'Fig2.png')), dpi=fset.dpi,
                bbox_inches='tight')

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()

    r2_value, mse_value, rmse_value, mae_value, mape_value = performance_evaluation('Prophet', meas, yhat)
    return r2_value, mse_value, rmse_value, mae_value, mape_value


def prophet_forecasting(data_file_name):
    # https://facebook.github.io/prophet/docs/quick_start.html
    start_time = time.time()
    df = pd.read_csv(data_file_name, parse_dates=['DT'])
    df.rename(columns={'DT': 'ds', 'Measurement': 'y'}, inplace=True)
    print('\nProphet dataframe YYYY-MM-DD HH:MM:SS')
    print(df)

    # Обучающая и тестовая выборки
    df_train = df[(df['ds'] >= datetime.datetime(2022, 12, 12)) & (df['ds'] < datetime.datetime(2022, 12, 29))]
    print('\nTrain dataset')
    print(df_train)

    df_test = df[(df['ds'] >= datetime.datetime(2022, 12, 29)) & (df['ds'] < datetime.datetime(2022, 12, 31))]
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
    xdat = forecast['ds']
    meas = df_test['y']
    yhat = forecast['yhat']
    yhat_lower = forecast['yhat_lower']
    yhat_upper = forecast['yhat_upper']

    r2_value, mse_value, rmse_value, mae_value, mape_value =\
        prophet_evaluation(xdat, meas, yhat, yhat_lower, yhat_upper)
    write_report(r2_value, mse_value, rmse_value, mae_value, mape_value, 'Prophet', True)
    write_report(r2_value, mse_value, rmse_value, mae_value, mape_value, 'Prophet')


if __name__ == '__main__':
    rep_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'report_2022_12_8__2023_1_1.csv'))
    dat_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'data.csv'))
    col_name = 'Influent'
    # Data Prepocessing
    data_preprocessing(rep_file_name, col_name, dat_file_name)
    # Prophet Forecasting
    prophet_forecasting(dat_file_name)
