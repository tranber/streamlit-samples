import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import altair as alt

DATA_URL = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
COUNTRY = "countriesAndTerritories"

st.title("Analysis of COVID 19 data")

today = date.today().strftime("%B %d, %Y %H:%M")

st.markdown(f"Done on {today}.")

@st.cache
def download_data():
    return pd.read_csv(DATA_URL)

text_loading = st.markdown("Loading data...")
data = download_data().copy()
text_loading.markdown(f"Data loaded, {data.size} records, "
    "from European Centre for Disease Prevention and Control")


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
    return df

ma = st.slider("Moving average:", min_value=1,
    max_value=40)

df_country = get_country_data(selected_country)

averaged_data = df_country[['cases']].rolling(ma).mean()
averaged_data.reset_index(inplace=True)

# some debug for now
#st.write(averaged_data.head())
x_scale = (
        alt.Scale(type="utc") 
    )
c = alt.Chart(averaged_data).mark_line().encode(
    alt.X("date:T", title="Date", scale=x_scale),
    alt.Y("cases", title="Nb cases"),
#    alt.Color("variable", title="", type="nominal"),
#    alt.Tooltip(["date", "cases", "variable"]),
)
#c = alt.generate_chart("line", averaged_data, 0, 0)

x = st.altair_chart(c, use_container_width=True)