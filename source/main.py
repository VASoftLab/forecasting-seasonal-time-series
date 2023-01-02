import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def data_preprocessing(filename, colname):
    start_time = time.time()

    # Чтение данных
    print(f'Reading data from {filename}')
    df = pd.read_csv(filename, parse_dates=['DT'])
    print('\nInitial dataframe')
    print(df.head())

    # Новый датафрейм
    new_df = df[['DT', colname]]
    # Установка индекса
    new_df = new_df.set_index('DT')
    # Передискретизация / Resample
    # W - неделя
    # D - календарный день
    # H - час
    # T - минута
    # S - секунда
    avg_df = pd.DataFrame(data=new_df[colname].resample('H').mean())
    # Изменяем заголовок столбца
    avg_df.columns = ['Value']
    print('\nResampled dataframe')
    print(avg_df.head())
    # Сохранить новый датафрейм в файл
    dir_name = os.path.dirname(filename)
    fil_name = os.path.join(dir_name, 'data.csv')
    avg_df.to_csv(fil_name)
    print(f'\nNew file saved to {fil_name}')
    print(f'\nExecution time: {(time.time() - start_time):3.2f} sec.')

    # Визуализация
    x = avg_df['Value']
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_dpi(150)
    ax.plot(x)
    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel('Measurement')
    plt.title('Initial Dataset')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'img', 'Fig1.png')), dpi=150)
    plt.show()


if __name__ == '__main__':
    # Подготовка данных
    datafile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'report_2022_12_8__2023_1_1.csv'))
    colname = 'Influent'
    data_preprocessing(datafile, colname)
