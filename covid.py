import streamlit as st
import datetime
import pandas as pd
import numpy as np
import altair as alt

DATA_URL = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
COUNTRY = "countriesAndTerritories"

st.title("Analysis of COVID 19 data")

today = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

st.markdown(f"Done on {today}.")

@st.cache
def download_data():
    return pd.read_csv(DATA_URL)

text_loading = st.markdown("Loading data...")
data = download_data().copy()
text_loading.markdown(f"Data loaded: {data.size} records, "
    "from European Centre for Disease Prevention and Control.")


data['date'] = pd.to_datetime(data['dateRep'], dayfirst=True)
data.sort_values(by='date', inplace=True)

st.sidebar.header("Options")
show_sample = st.sidebar.checkbox("Show data sample")
if show_sample:
    st.subheader("Sample Data:")
    st.write(data.sample(5))

st.sidebar.markdown("License Information:\n"
    + "Data downloaded from "
    + "[ECDC](https://www.ecdc.europa.eu/en/copyright)")
st.header("Country Analysis")
countries = data[COUNTRY].unique()
countries.sort()
country_index = int(np.where(countries =='France')[0][0])

selected_country = st.selectbox("Select Countries:",
    countries.tolist(),
    index=country_index)

@st.cache
def get_country_data(country:str):
    df = data[data[COUNTRY] == country]
    df.set_index('date', inplace=True)
    df['relative'] = df['cases'] / df['popData2018']
    return df

df_country = get_country_data(selected_country)

population = df_country.iloc[0].popData2018 
st.markdown(f"{selected_country} population in 2018: {population:,.0f}")

ma = st.slider("Moving average:", min_value=1,
    max_value=40)
is_relative = st.sidebar.checkbox("Use population relative figures")


column = 'cases' if not is_relative else 'relative'
averaged_data = df_country[[column]].rolling(ma).mean()
averaged_data.reset_index(inplace=True)

# some debug for now
if show_sample:
    st.write(averaged_data.head())

c = alt.Chart(averaged_data).mark_line().encode(
    alt.X("date:T", title="Date"),
    alt.Y("cases", title="Nb cases") if not is_relative
        else  alt.Y("relative:Q", axis=alt.Axis(format='%'), title="% Cases"),
    alt.Color("country", title="FR", type="nominal"),
    alt.Tooltip(["date", column]),
)
#c = alt.generate_chart("line", averaged_data, 0, 0)

x = st.altair_chart(c, use_container_width=True)