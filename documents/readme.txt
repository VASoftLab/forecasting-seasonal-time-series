Evaluation of the Machine Learning Techniques for Forecasting the Seasonal Time Series

1. Загрузить данные и преобразовать из в DateFrame
Учитывать формат даты / времени
https://facebook.github.io/prophet/docs/quick_start.html
The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

Ссылки Prophet
https://facebook.github.io/prophet/docs/quick_start.html
https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/
