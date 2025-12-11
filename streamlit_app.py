
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import pickle
from pathlib import Path
import time

# Пути
RESULTS_DIR = Path("/content/results")

# Настройка страницы
st.set_page_config(
    page_title="Salary Prediction AI - Результаты",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Функция для проверки существования файлов
def check_files_exist():
    """Проверяет, существуют ли необходимые файлы"""
    required_files = [
        RESULTS_DIR / "cleaned_data.csv",
        RESULTS_DIR / "model_results.csv",
        RESULTS_DIR / "analysis_report.json"
    ]

    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path.name)

    return missing_files

# Загрузка данных с проверкой
@st.cache_resource
def load_results():
    try:
        # Проверяем существование файлов
        missing_files = check_files_exist()
        if missing_files:
            st.error(f"Отсутствуют файлы: {', '.join(missing_files)}")
            return None

        # Загружаем очищенные данные
        cleaned_df = pd.read_csv(RESULTS_DIR / "cleaned_data.csv")

        # Загружаем результаты моделей
        model_results = pd.read_csv(RESULTS_DIR / "model_results.csv")

        # Загружаем отчет анализа
        with open(RESULTS_DIR / "analysis_report.json", 'r') as f:
            analysis_report = json.load(f)

        return {
            'cleaned_df': cleaned_df,
            'model_results': model_results,
            'analysis_report': analysis_report
        }
    except Exception as e:
        st.error(f"Ошибка загрузки результатов: {str(e)}")
        return None

# Визуализатор (оставляем без изменений)
class SalaryVisualizer:
    @staticmethod
    def plot_salary_distribution(df):
        """Распределение зарплат"""
        if 'Salary' not in df.columns:
            return None

        fig = px.histogram(
            df, x='Salary',
            title='Распределение зарплат',
            nbins=50,
            color_discrete_sequence=['#3498db'],
            opacity=0.85,
            marginal='box'
        )
        fig.update_layout(
            xaxis_title="Зарплата (₹)",
            yaxis_title="Количество сотрудников",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_education_vs_salary(df):
        """Зарплата по образованию"""
        if not all(col in df.columns for col in ['Education Level', 'Salary']):
            return None

        fig = px.box(
            df, x='Education Level', y='Salary',
            title='Зарплата по уровню образования',
            color='Education Level',
            points='all'
        )
        fig.update_layout(
            xaxis_title="Образование",
            yaxis_title="Зарплата (₹)",
            template="plotly_white",
            showlegend=False,
            xaxis_tickangle=45
        )
        return fig

    @staticmethod
    def plot_job_title_analysis(df):
        """Анализ по должностям"""
        if 'Job Title' not in df.columns:
            return None

        job_counts = df['Job Title'].value_counts().head(15)
        fig = px.bar(
            x=job_counts.index,
            y=job_counts.values,
            title='Топ-15 должностей по количеству',
            labels={'x': 'Должность', 'y': 'Количество'},
            color=job_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title="Должность",
            yaxis_title="Количество",
            template="plotly_white",
            xaxis_tickangle=45
        )
        return fig

    @staticmethod
    def plot_salary_by_job_title(df):
        """Средняя зарплата по должностям"""
        if not all(col in df.columns for col in ['Job Title', 'Salary']):
            return None

        job_salary = df.groupby('Job Title')['Salary'].agg(['mean', 'count']).reset_index()
        job_salary = job_salary.sort_values('mean', ascending=False).head(10)

        fig = px.bar(
            job_salary,
            x='Job Title',
            y='mean',
            title='Топ-10 должностей по средней зарплате',
            labels={'mean': 'Средняя зарплата (₹)', 'Job Title': 'Должность'},
            color='mean',
            color_continuous_scale='Plasma',
            text='mean'
        )
        fig.update_traces(
            texttemplate='₹%{text:.0f}',
            textposition='outside'
        )
        fig.update_layout(
            xaxis_title="Должность",
            yaxis_title="Средняя зарплата (₹)",
            template="plotly_white",
            xaxis_tickangle=45
        )
        return fig

    @staticmethod
    def plot_age_vs_salary(df):
        """Зависимость зарплаты от возраста"""
        if not all(col in df.columns for col in ['Age', 'Salary']):
            return None

        fig = px.scatter(
            df, x='Age', y='Salary',
            title='Зависимость зарплаты от возраста',
            color='Education Level' if 'Education Level' in df.columns else None,
            trendline='ols',
            opacity=0.7
        )
        fig.update_layout(
            xaxis_title="Возраст (лет)",
            yaxis_title="Зарплата (₹)",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_gender_analysis(df):
        """Анализ по полу"""
        if 'Gender' not in df.columns:
            return None

        gender_stats = df.groupby('Gender')['Salary'].agg(['mean', 'count']).reset_index()
        fig = px.bar(
            gender_stats,
            x='Gender',
            y='mean',
            title='Средняя зарплата по полу',
            labels={'mean': 'Средняя зарплата (₹)', 'Gender': 'Пол'},
            color='Gender',
            text='mean'
        )
        fig.update_traces(
            texttemplate='₹%{text:.0f}',
            textposition='outside'
        )
        fig.update_layout(
            xaxis_title="Пол",
            yaxis_title="Средняя зарплата (₹)",
            template="plotly_white",
            showlegend=False
        )
        return fig

    @staticmethod
    def plot_experience_distribution(df):
        """Распределение опыта работы"""
        if 'Years of Experience' not in df.columns:
            return None

        fig = px.histogram(
            df, x='Years of Experience',
            title='Распределение опыта работы',
            nbins=30,
            color_discrete_sequence=['#e74c3c'],
            opacity=0.85,
            marginal='rug'
        )
        fig.update_layout(
            xaxis_title="Опыт работы (лет)",
            yaxis_title="Количество",
            template="plotly_white"
        )
        return fig

# Главная функция
def main():
    st.title("Salary Prediction AI - Результаты анализа")
    st.markdown("""
    ### Все операции были выполнены в Google Colab. На этой странице отображаются результаты.
    **Датасет:** mohithsairamreddy/salary-data
    **Колонки:** Age, Gender, Education Level, Job Title, Years of Experience, Salary
    **Валюта:** Индийские рупии (₹)
    """)

    # Проверяем существование файлов
    missing_files = check_files_exist()

    if missing_files:
        st.warning("""
        ### Результаты анализа не найдены или неполные!

        Пожалуйста, сначала запустите пайплайн анализа в Google Colab:

        1. Выполните пайплайн анализа во 2-й ячейке
        2. Дождитесь завершения всех этапов анализа
        3. Убедитесь, что появилось сообщение "Пайплайн успешно завершен!"
        4. Затем обновите эту страницу (нажмите F5 или кнопку обновления в браузере)

        **Отсутствуют файлы:**
        """)

        for file in missing_files:
            st.write(f"- `{file}`")

        st.info("""
        **Статус выполнения:**
        1. Настройка Kaggle API (ячейка 1)
        2. Запуск пайплайна анализа (ячейка 2) - выполняется...
        3. Запуск Streamlit (ячейка 3) - ожидание результатов
        """)

        # Кнопка для проверки готовности
        if st.button("Проверить готовность результатов"):
            st.rerun()

        return

    # Загружаем результаты
    results = load_results()

    if results is None:
        st.error("Не удалось загрузить результаты анализа")
        return

    cleaned_df = results['cleaned_df']
    model_results = results['model_results']
    analysis_report = results['analysis_report']
    visualizer = SalaryVisualizer()

    # Сайдбар с навигацией
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выберите раздел:",
        ["Главная", "Анализ данных", "Результаты ML", "Визуализация", "Прогнозирование"]
    )

    # Главная страница
    if page == "Главная":
        st.header("Общая информация")

        # Статус Kaggle
        if analysis_report.get('username'):
            st.success(f"Kaggle API настроен для пользователя: **{analysis_report['username']}**")

        # Основные метрики
        cols = st.columns(4)
        with cols[0]:
            st.metric("Очищено строк", cleaned_df.shape[0])
        with cols[1]:
            st.metric("Колонок", cleaned_df.shape[1])
        with cols[2]:
            missing_total = cleaned_df.isnull().sum().sum()
            st.metric("Пропусков", missing_total)
        with cols[3]:
            best_model = analysis_report['best_model']['Model']
            st.metric("Лучшая модель", best_model)

        # Предпросмотр данных
        with st.expander("Предпросмотр очищенных данных (первые 10 строк)"):
            st.dataframe(cleaned_df.head(10), use_container_width=True)

        # Этапы выполнения
        st.markdown(f"""
        ### Этапы выполнения:
        1. **Настройка Kaggle API** в консоли Colab ✓
        2. **Загрузка датасета** salary-data с Kaggle ✓
        3. **Загрузка и анализ данных** ✓
        4. **Анализ данных до очистки** ✓
        5. **Автоматическая очистка данных** ✓
        6. **Анализ данных после очистки** ✓
        7. **Обучение ML моделей на очищенных данных** ✓
        8. **Визуализация результатов на очищенных данных** ✓
        """)

    # Анализ данных
    elif page == "Анализ данных":
        st.header("Анализ данных")

        tab1, tab2 = st.tabs(["До очистки", "После очистки"])

        with tab1:
            st.subheader("Анализ данных до очистки")
            report_before = analysis_report['report_before']

            cols = st.columns(4)
            with cols[0]:
                st.metric("Строк", report_before['basic_info']['rows'])
            with cols[1]:
                st.metric("Пропусков", report_before['missing_values']['total'])
            with cols[2]:
                st.metric("Дубликатов", report_before['quality_metrics']['duplicates'])
            with cols[3]:
                st.metric("Качество", f"{report_before['quality_metrics']['score']}%")

            # Детальная статистика зарплаты
            if 'salary_stats' in report_before:
                with st.expander("Статистики зарплаты до очистки"):
                    st.json(report_before['salary_stats'])

        with tab2:
            st.subheader("Анализ данных после очистки")
            report_after = analysis_report['report_after']

            cols = st.columns(4)
            with cols[0]:
                st.metric("Строк", report_after['basic_info']['rows'])
            with cols[1]:
                st.metric("Пропусков", report_after['missing_values']['total'])
            with cols[2]:
                st.metric("Дубликатов", report_after['quality_metrics']['duplicates'])
            with cols[3]:
                st.metric("Качество", f"{report_after['quality_metrics']['score']}%")

            # Сравнение
            st.subheader("Сравнение до/после очистки")
            if 'salary_stats' in report_before and 'salary_stats' in report_after:
                comp_cols = st.columns(3)
                with comp_cols[0]:
                    diff_score = report_after['quality_metrics']['score'] - report_before['quality_metrics']['score']
                    st.metric("Качество",
                             f"{report_after['quality_metrics']['score']}%",
                             f"{diff_score:+.1f}%")
                with comp_cols[1]:
                    diff_missing = report_before['missing_values']['total'] - report_after['missing_values']['total']
                    st.metric("Пропуски удалено",
                             f"{diff_missing}")
                with comp_cols[2]:
                    diff_salary = report_after['salary_stats']['mean'] - report_before['salary_stats']['mean']
                    st.metric("Средняя зарплата",
                             f"₹{report_after['salary_stats']['mean']:,.0f}",
                             f"₹{diff_salary:,.0f}")

    # Результаты ML
    elif page == "Результаты ML":
        st.header("Результаты обучения моделей")

        best_model = model_results.iloc[0]

        st.info(f"**Лучшая модель (обучена на очищенных данных):** {best_model['Model']}")
        st.info(f"**R² score:** {best_model['R²']}")
        st.info(f"**RMSE:** {best_model['RMSE']}")
        st.info(f"**MAE:** {best_model['MAE']}")

        st.markdown("""
        **Объяснение метрик:**
        - **R² score:** Показывает, насколько хорошо модель объясняет дисперсию данных (чем ближе к 1, тем лучше)
        - **RMSE (Root Mean Square Error):** Средняя ошибка предсказания
        - **MAE (Mean Absolute Error):** Средняя абсолютная ошибка
        """)

        # Таблица результатов
        st.subheader("Результаты всех моделей")
        st.dataframe(
            model_results.style.format({
                'R²': '{:.4f}',
                'RMSE': '{:.2f}',
                'MAE': '{:.2f}'
            }).background_gradient(subset=['R²'], cmap='RdYlGn'),
            use_container_width=True
        )

        # Сохранение результатов
        with st.expander("Скачать результаты"):
            col1, col2 = st.columns(2)
            with col1:
                csv = model_results.to_csv(index=False)
                st.download_button(
                    label="Скачать результаты моделей (CSV)",
                    data=csv,
                    file_name="model_results.csv",
                    mime="text/csv"
                )
            with col2:
                json_str = json.dumps(analysis_report, indent=2)
                st.download_button(
                    label="Скачать отчет анализа (JSON)",
                    data=json_str,
                    file_name="analysis_report.json",
                    mime="application/json"
                )

    # Визуализация
    elif page == "Визуализация":
        st.header("Визуализация данных")

        # Выбор графиков
        st.subheader("Выберите графики для отображения")

        col1, col2 = st.columns(2)
        with col1:
            show_salary = st.checkbox("Распределение зарплат", True)
            show_education = st.checkbox("Зарплата по образованию", True)
            show_jobs = st.checkbox("Анализ должностей", True)
            show_top_jobs = st.checkbox("Топ-10 должностей по зарплате", True)

        with col2:
            show_age = st.checkbox("Зависимость зарплаты от возраста", True)
            show_gender = st.checkbox("Анализ по полу", True)
            show_experience = st.checkbox("Распределение опыта работы", True)

        # Отображение графиков
        if show_salary:
            fig = visualizer.plot_salary_distribution(cleaned_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if show_education:
            fig = visualizer.plot_education_vs_salary(cleaned_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if show_jobs:
            fig = visualizer.plot_job_title_analysis(cleaned_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if show_top_jobs:
            fig = visualizer.plot_salary_by_job_title(cleaned_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if show_age:
            fig = visualizer.plot_age_vs_salary(cleaned_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if show_gender:
            fig = visualizer.plot_gender_analysis(cleaned_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if show_experience:
            fig = visualizer.plot_experience_distribution(cleaned_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Прогнозирование
    elif page == "Прогнозирование":
        st.header("Прогнозирование зарплаты")

        st.info(f"**Используется лучшая модель:** {analysis_report['best_model']['Model']}")

        # Форма ввода
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.slider(
                    "Возраст",
                    int(cleaned_df['Age'].min()) if 'Age' in cleaned_df.columns else 18,
                    int(cleaned_df['Age'].max()) if 'Age' in cleaned_df.columns else 70,
                    int(cleaned_df['Age'].median()) if 'Age' in cleaned_df.columns else 30,
                    help="Возраст сотрудника"
                )

                gender_options = sorted(cleaned_df['Gender'].unique().tolist()) if 'Gender' in cleaned_df.columns else ["Male", "Female", "Other"]
                gender = st.selectbox(
                    "Пол",
                    gender_options,
                    help="Пол сотрудника"
                )

                education_options = sorted(cleaned_df['Education Level'].unique().tolist()) if 'Education Level' in cleaned_df.columns else ["High School", "Bachelor'S", "Master'S", "Phd"]
                education = st.selectbox(
                    "Образование",
                    education_options,
                    help="Уровень образования"
                )

            with col2:
                job_options = sorted(cleaned_df['Job Title'].unique().tolist()) if 'Job Title' in cleaned_df.columns else ["Manager", "Engineer", "Analyst", "Developer", "Director"]
                job_title = st.selectbox(
                    "Должность",
                    job_options,
                    help="Должность сотрудника"
                )

                experience = st.slider(
                    "Опыт работы (лет)",
                    int(cleaned_df['Years of Experience'].min()) if 'Years of Experience' in cleaned_df.columns else 0,
                    int(cleaned_df['Years of Experience'].max()) if 'Years of Experience' in cleaned_df.columns else 40,
                    int(cleaned_df['Years of Experience'].median()) if 'Years of Experience' in cleaned_df.columns else 5,
                    help="Опыт работы в годах"
                )

            submitted = st.form_submit_button("Рассчитать зарплату", type="primary")

            if submitted:
                # Простое прогнозирование на основе данных
                if 'Salary' in cleaned_df.columns:
                    # Фильтруем данные по параметрам
                    filtered = cleaned_df[
                        (cleaned_df['Age'].between(age-3, age+3)) &
                        (cleaned_df['Gender'] == gender) &
                        (cleaned_df['Education Level'] == education) &
                        (cleaned_df['Job Title'] == job_title) &
                        (cleaned_df['Years of Experience'].between(experience-2, experience+2))
                    ]

                    if len(filtered) > 0:
                        predicted_salary = filtered['Salary'].mean()
                        count_matches = len(filtered)
                        st.success(f"Найдено {count_matches} похожих записей в датасете")
                    else:
                        # Если нет точных совпадений, используем логику на основе средних
                        base_salary = cleaned_df['Salary'].mean()

                        # Корректировки
                        age_factor = 1 + (age - 30) * 0.015  # +1.5% за каждый год после 30
                        exp_factor = 1 + experience * 0.025  # +2.5% за каждый год опыта

                        # Корректировка по образованию
                        edu_factors = {
                            "High School": 0.8,
                            "Bachelor'S": 1.0,
                            "Master'S": 1.25,
                            "Phd": 1.4
                        }
                        edu_factor = edu_factors.get(education, 1.0)

                        # Корректировка по должности
                        job_salaries = cleaned_df.groupby('Job Title')['Salary'].mean()
                        if job_title in job_salaries.index:
                            job_factor = job_salaries[job_title] / base_salary
                        else:
                            job_factor = 1.0

                        predicted_salary = base_salary * age_factor * exp_factor * edu_factor * job_factor
                        st.info("Точных совпадений не найдено. Использована приближенная оценка на основе статистики.")

                    salary_mean = cleaned_df['Salary'].mean()
                    salary_median = cleaned_df['Salary'].median()

                    st.success(f"### Прогнозируемая зарплата: **₹{predicted_salary:,.0f}**")

                    # Дополнительная информация
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Средняя по датасету", f"₹{salary_mean:,.0f}")
                    with col2:
                        diff = predicted_salary - salary_mean
                        st.metric("Разница от среднего", f"₹{diff:,.0f}")
                    with col3:
                        perc = (diff / salary_mean * 100) if salary_mean > 0 else 0
                        st.metric("Процент разницы", f"{perc:.1f}%")

                    # Доверительный интервал
                    rmse = analysis_report['best_model']['RMSE']
                    with st.expander("Оценка точности прогноза"):
                        st.write(f"**На основе метрик лучшей модели:**")
                        st.write(f"- RMSE (Среднеквадратичная ошибка): {rmse:,.0f}")
                        st.write(f"- Доверительный интервал (±2×RMSE): {predicted_salary - 2*rmse:,.0f} - {predicted_salary + 2*rmse:,.0f}")
                        st.write(f"- Погрешность: ±{2*rmse:,.0f}")
                else:
                    st.warning("Невозможно сделать прогноз. Данные о зарплате не найдены.")

if __name__ == "__main__":
    main()
