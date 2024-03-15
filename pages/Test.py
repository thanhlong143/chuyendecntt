import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time
from streamlit_extras.metric_cards import style_metric_cards
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objs as go

#from query import *

st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
st.header("Phân tích học tập, Điểm, Tình trạng sinh viên & Dự đoán")

theme_plotly = None 

with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#result = view_all_data()
#df=pd.DataFrame(result,columns=["Policy","Expiry","Location","Nganh","Region","SoTCTL","Construction","XepLoaiTN","Earthquake","Flood","DiemTBCHe4","id"])

df=pd.read_excel('./datasets/dot thang 8_2023_Danh sach SV duoc cong nhan TN ĐHCQ_sv.xlsx', sheet_name='CQ')
df2=pd.read_excel("./datasets/Danh_sach_du_kien_HBKKHT_HK II-2022-2023.xlsx")
# print((df["MSSV"].nunique()))
# print((df.loc[df["DiemTBCHe4"] > 3.2]).shape)

print(df2['LoaiHB'])

