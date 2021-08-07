import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import seaborn as sns
from apps.model import PcmProperties, EutecticMixture

@st.cache(allow_output_mutation=True)
def load_data():
    df1 = pd.read_excel('./data/pcmdata.xlsx')
    df2 = pd.read_csv('./data/eutectic_enthalpy.csv')
    return df1,df2


def app():
    df_data,df_enthalpy=load_data()

    with st.beta_expander("Recommended component combinations"):
        def sort_nearby(x):
            return min(abs(x-selected_melting_T),abs(selected_melting_T-x))

        selected_melting_T = st.number_input("Enter temperature in degrees celcius",help="Select temperature")
        df_enthalpy['d'] =df_enthalpy['TE'].apply(sort_nearby)
        d =df_enthalpy.sort_values(by='d').head(8)
        st.table(d.head(8))
    with st.beta_expander("Get binary mixture properties"):
        cols = st.beta_columns(2)
        compA =cols[0].selectbox("Select A",df_data['pcm'].unique())
        compB =cols[1].selectbox("Select B",df_data['pcm'].unique())
        A = PcmProperties(df_data[df_data['pcm']==compA].values.tolist()[0])
        B = PcmProperties(df_data[df_data['pcm']==compB].values.tolist()[0])
        mixture = EutecticMixture(A,B)
        TE, xE = mixture.eutectic_properties()
        HE = mixture.enthalpy()
        st.pyplot(mixture.plot_temp_AB())
        st.markdown(f"Eutect temperature of the mixture {round(TE,2)} K")
        st.markdown(f"Eutect mole fraction of A {xE}")
        #left,right = st.beta_columns(2)
        st.pyplot(mixture.plot_entropy())
        st.pyplot(mixture.plot_enthalpy())
        st.markdown(f"Eutect mixture heat of fusion {HE} J/mol.K")


if __name__=='__main__':
    app()
