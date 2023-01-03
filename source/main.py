import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet


def data_preprocessing(report_file_name, colimn_name, data_file_name):
    start_time = time.time()

    # Чтение данных
    print(f'Reading data from {report_file_name}')
    df = pd.read_csv(report_file_name, parse_dates=['DT'])
    print('\nInitial dataframe')
    print(df.head())

    # Новый датафрейм
    new_df = df[['DT', colimn_name]]
    # Установка индекса
    new_df = new_df.set_index('DT')
    new_df = new_df[(new_df.index >= datetime.datetime(2022, 12, 12)) & (new_df.index < datetime.datetime(2022, 12, 31))]
    # Передискретизация / Resample
    # W - неделя
    # D - календарный день
    # H - час
    # T - минута
    # S - секунда
    avg_df = pd.DataFrame(data=new_df[colimn_name].resample('H').mean())
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
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_dpi(150)
    ax.plot(values)
    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel('Measurement')
    plt.title('Initial dataset')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'img', 'Fig1.png')), dpi=150)
    plt.show()


def prophet_evaluation(xdat, meas, yhat, yhat_lower, yhat_upper):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_dpi(150)
    ax.plot(xdat, yhat, '-')
    ax.fill_between(xdat, yhat_lower, yhat_upper, alpha=0.2)
    ax.plot(xdat, meas, 'o', color='tab:brown', markersize=5)
    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel('Measurement')
    plt.title('Prophet evaluation')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'img', 'Fig2.png')), dpi=150)
    plt.show()


def prophet_forecasting(data_file_name):
    # https://facebook.github.io/prophet/docs/quick_start.html
    start_time = time.time()
    df = pd.read_csv(data_file_name, parse_dates=['DT'])
    df.rename(columns={'DT': 'ds', 'Measurement': 'y'}, inplace=True)
    print('\nProphet dataframe YYYY-MM-DD HH:MM:SS')
    print(df)

    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=48, freq='60min', include_history=False)
    print('\nProphet future')
    print(future)

    forecast = m.predict(future)
    print('\nforecast')
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    print(f'\nExecution time: {(time.time() - start_time):3.2f} sec.')
    xdat = forecast['ds']
    meas = forecast['yhat']
    yhat = forecast['yhat']
    yhat_lower = forecast['yhat_lower']
    yhat_upper = forecast['yhat_upper']
    prophet_evaluation(xdat, meas, yhat, yhat_lower, yhat_upper)


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    rep_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'report_2022_12_8__2023_1_1.csv'))
    dat_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'data.csv'))
    col_name = 'Influent'
    data_preprocessing(rep_file_name, col_name, dat_file_name)
    prophet_forecasting(dat_file_name)
