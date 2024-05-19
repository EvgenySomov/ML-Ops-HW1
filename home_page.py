import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px

import xgboost as xgb


def home_page():
    st.header('1. Описание проекта:')

    st.markdown('''
    Данная работа представляет собой 1 домашнее задание ML Ops :red[МТС ШАД].\n
    **Выполнил**: Сомов Е.А.''')

    with st.popover("Дополнительная информация ❓"):
        st.markdown('''**Основная задача:** Успешно применить навыки контейнеризации с помощью :blue[Docker],
        поместив небольшой web интерфейс по предсказыванию оттока клиента на обученной модели.''')

    st.divider()
    st.header('2. Загрузка данных:')
    uploaded_file = st.file_uploader("Воспользуйтесь формой загрузки .csv файла", type=['csv'])

    preprocessing_test_data_frame = None
    test_data_frame = None
    features = []

    if uploaded_file is not None:
        try:
            test_data_frame = pd.read_csv(uploaded_file)
            preprocessing_test_data_frame, features = preprocessing(test_data_frame)

            st.success('Данные успешно загружены!')
        except:
            st.warning('Произошла ошибка, проверьте загружаемые данные!')

    if preprocessing_test_data_frame is not None:
        st.divider()
        st.header('3. Предсказание модели:')

        loaded_bst = xgb.Booster()
        loaded_bst.load_model('xgboost_model.json')
        dtest = xgb.DMatrix(preprocessing_test_data_frame.values)

        threshold = 0.32
        test_data_frame['predict_proba'] = loaded_bst.predict(dtest)
        test_data_frame['preds'] = (test_data_frame['predict_proba'] > threshold).astype(int)

        csv_file = test_data_frame[['client_id', 'preds']].to_csv(index=False).encode("utf-8")

        st.success('Submission готов!')

        download_col, popover_col = st.columns(2)
        with popover_col.popover("Дополнительная информация ❓"):
            st.markdown('''**Модель:** Испольюуется модель XGBoosting, обученная с помощью Optuna''')

        download_col.download_button(
            label="Скачать Submission",
            data=csv_file,
            file_name="submission.csv",
            mime="text/csv",
        )

        st.divider()
        st.header('4. Дополнительная информация:')

        expander_json = st.expander("Топ-5 feature importances:")
        expander_plot = st.expander("График* плотности распределения предсказанных моделью скоров:")

        feature_importances = loaded_bst.get_score(importance_type='weight')
        sorting_features = pd.Series(feature_importances).sort_values(ascending=False)
        ind_sort = sorting_features.index.to_numpy()

        sorting_features.index = features[np.array([int(feat[1:]) for feat in ind_sort])]
        expander_json.write(sorting_features[:5].to_dict())

        json_file = sorting_features[:5].to_json().encode("utf-8")

        expander_json.download_button(
            label="Скачать feature importances",
            data=json_file,
            file_name="feature_importances_5.json",
            mime="text/json",
        )

        fig = px.histogram(test_data_frame, x="predict_proba", nbins=20)
        fig.update_layout(
            title="Распределение вероятностей предсказаний",
            xaxis_title="Вероятность предсказания",
            yaxis_title="Количество",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',  # Прозрачный фон
            paper_bgcolor='rgba(255,255,255,255)',  # Белый фон рамки
            margin=dict(l=10, r=10, t=50, b=50)  # Поля вокруг графика
        )

        expander_plot.plotly_chart(fig, use_container_width=True)
        expander_plot.text('*Чтобы скачать график, воспользуйтесь встроенной в него панелью.')


def preprocessing(data: pd.DataFrame):
    data = data.copy()
    data['Сумма_пропусков'] = data.isna().sum(axis=1)
    data = data[['сумма', 'частота_пополнения',
                 'частота', 'on_net',
                 'сегмент_arpu', 'объем_данных',
                 'продукт_1', 'продукт_2',
                 'pack_freq', 'зона_1',
                 'зона_2', 'mrg_',
                 'секретный_скор', 'доход']]

    data['Сумма_пропусков'] = data.isna().sum(axis=1)
    data['Пропуск_суммы'] = np.where(data['сумма'].isna(), 1, 0)
    data['сумма'] = data['сумма'].fillna(-1)
    data['частота_пополнения'] = data['частота_пополнения'].fillna(-1)
    data['частота'] = data['частота'].fillna(-1)
    data['доход'] = data['доход'].fillna(-1)
    data['on_net'] = data['on_net'].fillna(-1)
    data['сегмент_arpu'] = data['сегмент_arpu'].fillna(0)
    data['объем_данных'] = data['объем_данных'].fillna(0)
    data['продукт_1'] = data['продукт_1'].fillna(-1)
    data['продукт_2'] = data['продукт_2'].fillna(-1)
    data['pack_freq'] = data['pack_freq'].fillna(-1)
    data['зона_1'] = data['зона_1'].fillna(-1)
    data['зона_2'] = data['зона_2'].fillna(-1)
    data['mrg_'] = data['mrg_'].astype(int)

    features = data.columns
    return data, features