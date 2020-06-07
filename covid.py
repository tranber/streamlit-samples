import streamlit as st
import datetime
import pandas as pd
import numpy as np
import altair as alt

DATA_URL = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
COUNTRY = "countriesAndTerritories"

st.image("images/covid19x100.jpeg")
st.title("Analysis of COVID 19 data")

today = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
st.markdown(f"Done on {today}.")

@st.cache
def download_data():
    return pd.read_csv(DATA_URL)

text_loading = st.markdown("Loading data...")
data = download_data().copy()
text_loading.markdown(f"Data loaded: {data.size:,.0f} records, "
    "from European Centre for Disease Prevention and Control.")


data['date'] = pd.to_datetime(data['dateRep'], dayfirst=True)
data[COUNTRY] = data[COUNTRY].str.replace('_', ' ')
data.sort_values(by='date', inplace=True)

st.sidebar.header("Options")
show_sample = st.sidebar.checkbox("Show data sample")
is_relative = st.sidebar.checkbox("Use population relative figures")
is_cumulative = st.sidebar.checkbox("Cumulative Sum")
log_scale = st.sidebar.checkbox("Use log scale")

st.sidebar.info("License Information:\n\n"
    + "Data downloaded from "
    + "[ECDC](https://www.ecdc.europa.eu/en/copyright)")


if show_sample:
    st.subheader("Sample Data:")
    st.write(data.sample(5))

st.header("Country Analysis")
countries = data[COUNTRY].unique()
countries.sort()

selected_countries = st.multiselect("Select Countries:",
    countries.tolist(),
    default=['France', 'Spain'])

series = st.radio("Select data to analyze:", ['cases', 'deaths'],
    format_func=str.capitalize)
ma = st.slider("Moving average:", min_value=1,
    max_value=40)

@st.cache
def get_country_data(country:str, series:str, is_relative:bool):
    df = data[data[COUNTRY] == country]
    df.set_index('date', inplace=True)
    if is_relative:
        # compute nb of case per million
        df[series] = df[series] / df['popData2018'] * 1_000_000
    df_country = df[[series]]
    return df_country


df_all:pd.DataFrame = None
for country in selected_countries:
    df_country = get_country_data(country, series, is_relative).copy(deep=True)
    if is_cumulative:
        averaged_data = df_country[[series]].cumsum()
        averaged_data = averaged_data.rolling(ma).mean()
    else:
        averaged_data = df_country[[series]].rolling(ma).mean()
    averaged_data['Country'] = country
    if df_all is None:
        df_all = averaged_data
    else:
        df_all = pd.concat([df_all, averaged_data])


# some debug for now
if show_sample:
    st.write("Sample Data")
    st.write(df_all.head())


df_all.reset_index(inplace=True)
cumul = (" (Cumulated)" if is_cumulative else "")
chart_title = (("Number of %s in Selected countries%s" % (series, cumul)) if not is_relative
    else ("Number of %s for 1M %s" % (series, cumul)))
c = alt.Chart(df_all, title=chart_title).mark_line().encode(
    x='date:T',
    y=(alt.Y(series,
        scale=alt.Scale(type=('symlog' if log_scale else 'linear')),
        axis=alt.Axis(format=',.0f',
                      title=(('Nb of %s per 1M habitant' % (series)) if is_relative else ('Nb. of %s'% (series)))),
        )),
    color='Country'
)

x = st.altair_chart(c, use_container_width=True)


