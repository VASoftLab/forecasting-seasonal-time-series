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
    new_df = new_df[(new_df.index >= datetime.datetime(2022, 12, 12)) &
                    (new_df.index < datetime.datetime(2022, 12, 31))]
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
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_dpi(150)
    ax.plot(values)
    ax.axvline(x=datetime.datetime(2022, 12, 29), color='g', alpha=0.5)
    ax.axvline(x=datetime.datetime(2022, 12, 31), color='g', alpha=0.5)

    ax.axvspan(datetime.datetime(2022, 12, 29),
               datetime.datetime(2022, 12, 31), color='g', alpha=0.2)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(datetime.datetime(2022, 12, 21), 8.5, "   Train Data   ",
            ha="center", va="center", size=18, bbox=bbox_props)
    ax.text(datetime.datetime(2022, 12, 30), 8.5, "Test Data",
            ha="center", va="center", size=18, bbox=bbox_props)

    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.title('Seasonal Time Series', fontweight="bold", fontsize=18)
    plt.xlabel('History Horizont', style='italic', fontsize=14)
    plt.ylabel('Measurement', style='italic', fontsize=14)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'img', 'Fig1.png')), dpi=150)
    plt.show()


def prophet_evaluation(xdat, meas, yhat, yhat_lower, yhat_upper):
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_dpi(150)
    ax.plot(xdat, yhat, '-')
    ax.plot(xdat, meas, 'o', color='tab:brown', markersize=5)
    ax.fill_between(xdat, yhat_lower, yhat_upper, alpha=0.2)
    ax.legend(['Predicted Value', 'Actual Value'], fontsize=14)

    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.title('Prophet Evaluation', fontweight="bold", fontsize=18)
    plt.xlabel('Forecast Horizont', style='italic', fontsize=14)
    plt.ylabel('Measurement', style='italic', fontsize=14)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
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
    prophet_evaluation(xdat, meas, yhat, yhat_lower, yhat_upper)


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    rep_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'report_2022_12_8__2023_1_1.csv'))
    dat_file_name = os.path.realpath(os.path.join(dir_name, '..', 'data', 'data.csv'))
    col_name = 'Influent'
    data_preprocessing(rep_file_name, col_name, dat_file_name)
    prophet_forecasting(dat_file_name)
