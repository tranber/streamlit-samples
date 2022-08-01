import streamlit as st
import datetime
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from typing import Sequence, List, Optional, Dict
from sklearn.cluster import KMeans
from collections import defaultdict
from enum import Enum

st.set_page_config(page_title='Covid data analysis',
    layout='wide', initial_sidebar_state='expanded')

DATA_URL = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
COUNTRY = "countriesAndTerritories"
POPULATION = 'popData2019'

class ChartType(Enum):
    ALTAIR = "Altair"
    PLOTLY = "Plotly"


@st.cache(ttl=60*60)
def get_original_data() -> pd.DataFrame:
    """Get the raw data, adding a date field, and removing the _ in country names
    """
    data = pd.read_csv(DATA_URL)
    # Adjust column names now that data are agreggated by week
    data.rename(columns={"cases_weekly": "cases", "deaths_weekly": "deaths"}, inplace=True)

    data['date'] = pd.to_datetime(data['dateRep'], dayfirst=True)
    data[COUNTRY] = data[COUNTRY].str.replace('_', ' ')
    data.sort_values(by='date', inplace=True)

    return data

@st.cache(ttl=60*60)
def get_countries_names():
    countries = get_original_data()[COUNTRY].unique()
    countries.sort()
    return countries

@st.cache(ttl=60*60)
def get_population_by_country() -> Dict[str, int]:
    """Get a map of country/population
    """
    countries = get_countries_names()
    data = get_original_data()
    population_for = {}
    for c in countries:
        try:
            population_for[c] = int(data[data[COUNTRY]==c][POPULATION].iloc[0])
        except ValueError:
            population_for[c] = -1
    return population_for


@st.cache
def get_country_data(data:pd.DataFrame, country:str, series:str, is_relative:bool):
    df = data[data[COUNTRY] == country]
    df.set_index('date', inplace=True)
    if is_relative:
        # compute nb of case per million
        df[series] = df[series] / df[POPULATION] * 1_000_000
    df_country = df[[series]]
    return df_country

#
# Try to print a human readable number
def human_format_number(n):
    formats = [(1e9, 999_500_000, 'B', 1), (1e6, 999_500, 'M', None), (1e3, 1e3, 'K', None)]
    for (divid, limit, letter, ndigits) in formats:
        if n >= limit:
            rounded = round(float(n) / divid, ndigits)
            return str(rounded) + letter
    return str(n)


# Prepare dataframe for plotting: extract countries, average, ...
def prepare_data(data:pd.DataFrame, countries:Sequence[str], series:str, ma:int, is_relative:bool, is_cumulative:bool,
        pivoting:bool=False) -> pd.DataFrame:
    df_all: pd.DataFrame = None
    for country in countries:
        df_country = get_country_data(data, country, series, is_relative).copy(deep=True)
        serie_data = df_country[[series]]
        if is_cumulative:
            serie_data = serie_data.cumsum()
        averaged_data = serie_data.rolling(ma).mean()
        if not pivoting:
            averaged_data['Country'] = country
        if df_all is None:
            df_all = averaged_data
        else:
            if pivoting:
                df_all[country] = averaged_data
            else:
                df_all = pd.concat([df_all, averaged_data])
    #if not pivoting:
    df_all.reset_index(inplace=True)
    return df_all



def page_country_analysis():
    # Header and load data

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("images/covid19x100.jpeg")
    with col2:
        st.title("Analysis of COVID 19 data")

    today = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
    st.markdown(f"Done on {today}.")

    text_loading = st.markdown("Loading data...")
    data = get_original_data()
    text_loading.markdown(f"Data loaded: {data.size:,.0f} records, "
        "from European Centre for Disease Prevention and Control.\n\n" +
        "Number of cases and deaths are cumulated by week.")
 
    if show_sample:
        st.subheader("Sample Data:")
        st.write(data.sample(5))

    # Select Countries
    st.header("Country Analysis")
    countries = get_countries_names()
    population_for = get_population_by_country()
    # st.write(population_for)

    selected_countries = st.multiselect("Select Countries:",
        countries.tolist(),
        default=['France', 'Spain', 'Italy'])

    selected_countries_pop = ', '.join(['%s (%s)' % (c, human_format_number(population_for[c])) for c in selected_countries])
    st.write(selected_countries_pop)

    df_all = prepare_data(data, selected_countries, series, ma, is_relative, is_cumulative, pivoting=False)

    if show_sample:
        st.write("Sample Data")
        st.write(df_all)

    # Configure graph
    cumul = (" (Cumulated)" if is_cumulative else "")
    chart_title = (("Number of %s in Selected countries%s" % (series, cumul)) if not is_relative
        else ("Number of %s for 1M %s" % (series, cumul)))

    if chart_type == ChartType.ALTAIR:
        c = alt.Chart(df_all, title=chart_title).mark_line().encode(
            x='date:T',
            y=(alt.Y(series,
                scale=alt.Scale(type=('symlog' if log_scale else 'linear')),
                axis=alt.Axis(format=',.0f',
                            title=(('Nb of %s per 1M habitant' % (series)) if is_relative else ('Nb. of %s'% (series)))),
                )),
            # Specify domain so that colors are fixed
            color=alt.Color('Country', scale=alt.Scale(domain=selected_countries))
        )
        x = st.altair_chart(c, use_container_width=True)
    else:
        country_cols = list(df_all.columns)
        country_cols.remove("date")
        fig = px.line(df_all, title=chart_title, color='Country',
            x="date", y=series,
            template='none')
        fig.update_traces(hovertemplate="<b>%{x|%a %B %d}</b><br>"
            + "%{y}")
        fig.update_layout(hovermode="closest")
        if log_scale:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)



# prepare data for clustering
@st.cache
def get_clustering_data(data, countries, series, ma, is_relative, is_cumulative):
    df_all = prepare_data(data, countries, series, ma, is_relative, is_cumulative)
    df_all.set_index('date', inplace=True)
    data_all : Optional[pd.DataFrame] = None
    for country in countries:
        cnt_data = df_all[df_all['Country'] == country][[series]]
        cnt_data.rename(columns={series:country}, inplace=True)
        if data_all is None:
            data_all = cnt_data
        else:
            data_all[country] = cnt_data[country]
    data_all = data_all.transpose()
    data_all.fillna(0, inplace=True)
    return data_all

@st.cache
def run_clustering(clus_data, nb_clusters):
    kmeans = KMeans(init='random', n_clusters=nb_clusters, n_init=10, max_iter=600)
    kmeans.fit(clus_data)
    return kmeans


def page_clustering_countries():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("images/covid19x100.jpeg")
    with col2:
        st.title("Clustering Countries Data")

    countries = get_countries_names()
    population_for = get_population_by_country()

    # Excluded countries - by default smaller than 100K
    default_excluded_countries = [k for (k,v) in population_for.items() if v < 100_000]
    default_excluded_countries.sort()
    excluded_countries = st.multiselect("Exclude Countries from clustering:",
        countries.tolist(),
        default=default_excluded_countries)

    excluded_countries_pop = ', '.join(['%s (%s)' % (c, human_format_number(population_for[c])) for c in excluded_countries])
    st.write(excluded_countries_pop)

    countries = countries.tolist()
    for x in excluded_countries:
        countries.remove(x)
 
    nb_clusters = st.slider('Number of clusters: ', min_value=1, max_value=10, value=7)

    clus_data = get_clustering_data(get_original_data(), countries, series, ma, is_relative, is_cumulative)
    run_msg = st.markdown("Running clustering...")
    kmeans = run_clustering(clus_data, nb_clusters)

    run_msg.markdown("Nb of iterations: " + str(kmeans.n_iter_))
    centroids = kmeans.cluster_centers_

    df_centro = pd.DataFrame()
    for i in range(0, nb_clusters):
        cluster_df = pd.DataFrame(data=centroids[i], columns=[series])
        cluster_df['Cluster'] = 'Cluster %d' % (i+1)
        if i == 0:
            df_centro = cluster_df
        else:
            df_centro = pd.concat([df_centro, cluster_df])
    df_centro.reset_index(inplace=True)
    cluster_names = ['Cluster %d' % (i+1) for i in range(0, nb_clusters)]


    if show_sample:
        st.write(df_centro)

    x_title = (('Nb of %s per 1M habitant' % (series)) if is_relative else ('Nb. of %s'% (series)))
    if chart_type == ChartType.ALTAIR:
        clus_chart = alt.Chart(df_centro, title='Clusters profiles').mark_line().encode(
            x='index',
            y=(alt.Y(series,
                scale=alt.Scale(type=('symlog' if log_scale else 'linear')),
                axis=alt.Axis(format=',.0f', title=x_title),
                )),
            # Specify domain so that colors are fixed
            color=alt.Color('Cluster', scale=alt.Scale(domain=cluster_names))
        )

        x2 = st.altair_chart(clus_chart, use_container_width=True)
    else:
        fig = px.line(df_centro, title='Clusters profiles', color='Cluster',
            x="index", y=series,
            template='none')
        fig.update_layout(hovermode="closest")
        if log_scale:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    cluster_indexes = kmeans.labels_
    clusters_countries = defaultdict(list)
    for i in range(0, len(countries)):
        clusters_countries[cluster_indexes[i]].append(countries[i])
    
    for i in range(0, nb_clusters):
        st.markdown("#### " + cluster_names[i])
        st.write(clusters_countries[i])



# Configures pages
PAGES = {
    "Display Countries Data": page_country_analysis,
    "Clustering Countries" : page_clustering_countries,
}

# Configure Sidebar
st.sidebar.header("Menu")
selected_page = st.sidebar.radio("", list(PAGES.keys()))

st.sidebar.header("Analysis type")
# Configure type of data
series = st.sidebar.radio("Select data to analyze:", ['cases', 'deaths'],
    format_func=str.capitalize)
ma = st.sidebar.slider("Moving average:", min_value=1,
    max_value=8, value=1)
is_relative = st.sidebar.checkbox("Use population relative figures", value=True)


st.sidebar.header("Options")
chart_type = st.sidebar.radio("Chart type:", [ChartType.ALTAIR, ChartType.PLOTLY],
    format_func=lambda x: x.value)
show_sample = st.sidebar.checkbox("Show data sample")
is_cumulative = st.sidebar.checkbox("Cumulative Sum")
log_scale = st.sidebar.checkbox("Use log scale")

st.sidebar.info("#### License Information: \n"
    + "Data downloaded from "
    + "[ECDC](https://www.ecdc.europa.eu/en/copyright)")


PAGES[selected_page]()


