Описание ETS метода
https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-ets.html




Evaluation of the Machine Learning Techniques for Forecasting the Seasonal Time Series

Полезная книга (можно взять теориию по некоторым методам - SARIMA, Holt-Winters)
https://otexts.com/fpp3/
Forecasting: Principles and Practice (3rd ed)
Rob J Hyndman and George Athanasopoulos
Monash University, Australia


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



SARIMAX
https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/
https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
https://analyticsindiamag.com/complete-guide-to-sarimax-in-python-for-time-series-modeling/

SARIMAX Introdaction
https://www.statsmodels.org/v0.13.0/examples/notebooks/generated/statespace_sarimax_stata.html
SARIMAX FAQ
https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_faq.html
How to use SARIMAX
https://www.kaggle.com/code/poiupoiu/how-to-use-sarimax/notebook


Kagle
https://www.kaggle.com/code/sadeght/arima-sarima-simple-clear-analysis
https://www.kaggle.com/code/poiupoiu/how-to-use-sarimax
Поиск параметров
https://www.kaggle.com/code/prashant111/arima-model-for-time-series-forecasting

https://pypi.org/project/pmdarima/

uncertainty interval
https://www.geeksforgeeks.org/how-to-calculate-confidence-intervals-in-python/
https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
6

Еще один метод
https://iopscience.iop.org/article/10.1088/1757-899X/407/1/012153

https://timeseriesreasoning.com/contents/holt-winters-exponential-smoothing/


Посмотреть. Статью. Анализ сезонных жанных
https://www.sciencedirect.com/science/article/pii/S2667096822000027
Посмотреть
https://www.sciencedirect.com/science/article/abs/pii/S157407060501013X


Пример реализации Holt-Winters
https://stackoverflow.com/questions/50785479/holt-winters-time-series-forecasting-with-statsmodels
https://num.pyro.ai/en/stable/examples/holt_winters.html

Описание метода Holt-Winters Exponential Smoothing
https://www.analyticsvidhya.com/blog/2021/08/holt-winters-method-for-time-series-analysis/
https://timeseriesreasoning.com/contents/holt-winters-exponential-smoothing/


https://stackoverflow.com/questions/70277316/how-to-take-confidence-interval-of-statsmodels-tsa-holtwinters-exponentialsmooth

ETSModel
https://www.statsmodels.org/dev/examples/notebooks/generated/ets.html


LTSM
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
https://curiousily.com/posts/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python/

ИСПОЛЬЗОВАТЬ ВО ВВЕДЕНИИ
https://curiousily.com/posts/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python/


XGBoost
https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost
