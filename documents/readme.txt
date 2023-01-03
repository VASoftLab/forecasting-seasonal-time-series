Evaluation of the Machine Learning Techniques for Forecasting the Seasonal Time Series

1. Загрузить данные и преобразовать из в DateFrame
Учитывать формат даты / времени
https://facebook.github.io/prophet/docs/quick_start.html
The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

Ссылки Prophet
https://facebook.github.io/prophet/docs/quick_start.html
https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/


Предварительная обработка
Прогнать через фильтр?
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html

Установка Prophet
https://pypi.org/project/prophet/
Устранение ошибки plotly
https://stackoverflow.com/questions/67371645/i-am-getting-error-in-plotly-module-of-fb-prophet


Примеры реализации Prophet
https://habr.com/ru/company/ods/blog/323730/
https://mlcourse.ai/book/topic09/topic9_part2_facebook_prophet.html

Добавить ссылку на статью
https://facebook.github.io/prophet/static/prophet_paper_20170113.pdf


TODO
Визуализацию предсказания выполнить с помощью функции
https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#sphx-glr-gallery-lines-bars-and-markers-fill-between-demo-py
