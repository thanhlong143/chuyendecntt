import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objs as go
#from query import *\

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Phân Tích Học Tập NEU",page_icon="",layout="wide")
st.header("Phân tích học tập, Điểm, Tình trạng sinh viên & Dự đoán")

theme_plotly = None

with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#result = view_all_data()
#df=pd.DataFrame(result,columns=["Policy","Expiry","Location","Nganh","Region","SoTCTL","Construction","XepLoaiTN","Earthquake","Flood","DiemTBCHe4","id"])

df=pd.read_excel('./datasets/dot thang 8_2023_Danh sach SV duoc cong nhan TN ĐHCQ_sv.xlsx', sheet_name='CQ')

#switcher
nganh=st.sidebar.multiselect(
    "Chọn Ngành",
    options=df["Nganh"].unique(),
    default=df["Nganh"].unique(),
)
chuyenNganh=st.sidebar.multiselect(
    "Chọn Chuyên Ngành",
    options=df["ChuyenNganh"].unique(),
    default=df["ChuyenNganh"].unique(),
)
lopChuyenNganh=st.sidebar.multiselect(
    "Chọn Lớp Chuyên Ngành",
    options=df["LopChuyenNganh"].unique(),
    default=df["LopChuyenNganh"].unique(),
)
df_selection=df.query(
    "Nganh==@nganh & ChuyenNganh==@chuyenNganh & LopChuyenNganh==@lopChuyenNganh"
)

custom_dict = {'Xuất sắc': 0, 'Giỏi': 1, 'Khá': 2, 'Trung bình khá':3, 'Trung bình':4} 

def Home():
    with st.expander("XEM CHI TIẾT DỮ LIỆU EXCEL"):
        showData=st.multiselect('Filter: ',df_selection.columns,default=["STT","MSSV","Ho","Ten","NgaySinh","SoTCTL","DiemTBCHe10","DiemTBCHe4","XepLoaiTN","LopChuyenNganh","ChuyenNganh","Nganh"])
        st.dataframe(df_selection[showData],use_container_width=True)
    #compute top analytics
    #total_SoSV = float(pd.Series(df_selection['MSSV']).sum())
    total_SoSV = df_selection['MSSV'].nunique()
    SoTCTL_mean = float(pd.Series(df_selection['SoTCTL']).mean())
    SoTCTL_median= float(pd.Series(df_selection['SoTCTL']).median()) 
    DiemTBCHe4 = float(pd.Series(df_selection['DiemTBCHe4']).mean())
    DiemTBCHe10_max = float(pd.Series(df_selection['DiemTBCHe10']).max())

    total1,total2,total3,total4,total5=st.columns(5,gap='small')
    with total1:
        # st.info('Tổng số sinh viên',icon="")
        # st.metric(label="Sum MSSV",value=f"{total_SoSV}")
        # st.info('Tổng số sinh viên',icon="")
        st.metric(label="Tổng số sinh viên",value=numerize(total_SoSV),help=f""" Tổng số sinh viên: {total_SoSV} """)
        style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow=True)

    with total2:
        # st.info('Số tín chỉ cao nhất',icon="")
        st.metric(label="Điểm hệ 10 cao nhất",value=f"{DiemTBCHe10_max:,.2f}")

    with total3:
        # st.info('Số tín chỉ trung bình',icon="")
        st.metric(label="Số tín chỉ trung bình",value=f"{SoTCTL_mean:,.0f}")

    with total4:
        # st.info('Số tín chỉ trung tâm',icon="")
        st.metric(label="Số tín chỉ trung tâm",value=f"{SoTCTL_median:,.0f}")

    with total5:
        # st.info('Điểm TBC Hệ 4',icon="")
        st.metric(label="Điểm TBC Hệ 4",value=numerize(DiemTBCHe4),help=f""" Điểm TBC Hệ 4: {DiemTBCHe4} """)
        style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow=True)

    #variable distribution Histogram
    with st.expander("Phân phối tần suất"):
        df.hist(figsize=(16,8),color='#898784', zorder=2, rwidth=0.9,legend = ['SoTCTL']);
        st.pyplot()

#graphs
def graphs():
    #total_SoSV=int(df_selection["SoTCTL"]).sum()
    #averageDiemTBCHe4=int(round(df_selection["DiemTBCHe4"]).mean(),2) 

    diemTBCHe4_Nganh=df_selection.groupby(by=["ChuyenNganh"])["DiemTBCHe4"].mean()

    fig_Nganh=px.line(
        diemTBCHe4_Nganh,
        x=diemTBCHe4_Nganh.index,
        y="DiemTBCHe4",
        orientation="v",
        title="<b> Điểm TBC Hệ 4 theo Chuyên Ngành </b>",
        color_discrete_sequence=["#0083b8"]*len(diemTBCHe4_Nganh),
        template="plotly_white",
        labels={ "ChuyenNganh": "Chuyên Ngành",  "DiemTBCHe4": "Điểm TBC Hệ 4" }
    )
    fig_Nganh.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=True))
    )
    #fig_Nganh.update_traces(textinfo='percent+label', textposition='inside')

    soSV_by_ChuyenNganh=(
        df_selection.groupby(by=["ChuyenNganh"]).count()[["MSSV"]].sort_values(by="MSSV")
    )
    fig_SoTCTL=px.bar(
        soSV_by_ChuyenNganh,
        x=soSV_by_ChuyenNganh.index,
        y="MSSV",
        orientation="v",
        title="<b> Số Lượng SV Chuyên Ngành </b>",
        color_discrete_sequence=["#0083B8"]*len(soSV_by_ChuyenNganh),
        template="plotly_white",
        labels = {"MSSV":"Số Lượng SV", "ChuyenNganh":"Chuyên Ngành"}
    )
    fig_SoTCTL.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor='#cecdcd'),   
        paper_bgcolor='rgba(0, 0, 0, 0)', 
        xaxis=dict(showgrid=True, gridcolor='#cecdcd'),
    )

    left,right,center=st.columns(3)
    left.plotly_chart(fig_Nganh,use_container_width=True)
    right.plotly_chart(fig_SoTCTL,use_container_width=True)
    
    with center:
        #pie chart
        # fig = px.pie(df_selection, values=df_selection.groupby(by=["XepLoaiTN"]).count()[["MSSV"]], names="XepLoaiTN", title='Xep Loai TN')
        # fig.update_layout(legend_title="_______________________", legend_y=0.9)
        # fig.update_traces(textinfo='percent+label', textposition='inside')
        # st.plotly_chart(fig, use_container_width=False, theme=theme_plotly)
        
        fig = go.Figure(
            data=[
                go.Pie(
                    labels = sorted(df_selection["XepLoaiTN"].unique()),
                    values=df_selection.groupby(by=["XepLoaiTN"])["MSSV"].count()
                )
            ],
            layout=go.Layout(
                title=go.layout.Title(text="<b> Số Lượng SV theo Xếp loại </b>"),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis=dict(showgrid=True, gridcolor='#cecdcd'),
                yaxis=dict(showgrid=True, gridcolor='#cecdcd'),
                #font=dict(color='#cecdcd'),
            )
        )
        fig.update_layout(legend_title="Xếp Loại", legend_y=0.9)
        fig.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

def Progressbar():
    st.markdown("""<style>.stProgress > div > div > div > div { background-image: linear-gradient(to right, #99ff99 , #FFFF00)}</style>""",unsafe_allow_html=True,)
    target=70
    rows, columns = (df_selection.loc[df_selection["DiemTBCHe4"] >= 3.2]).shape
    total = df_selection["MSSV"].nunique()
    percent=round((rows/total*100))
    mybar=st.progress(0)

    if percent>=target:
        st.subheader("Mục Tiêu Hoàn Thành!")
    else:
        st.write("Tỉ lệ ",percent, "% " ,"sinh viên Xuất Sắc và Giỏi trên tổng số", (format(total, 'd')), "Sinh viên")
        for percent_complete in range(percent):
            time.sleep(0.1)
            mybar.progress(percent_complete+1,text=" Phần Trăm Mục Tiêu")

#menu bar
def sideBar():
    with st.sidebar:
        selected=option_menu(
            menu_title="Phân Tích Học Tập",
            options=["Trang Chủ","Tiến Trình"],
            icons=["house","eye"],
            menu_icon="cast",
            default_index=0
        )
    if selected=="Trang Chủ":
        #st.subheader(f"Page: {selected}")
        Home()
        graphs()
    if selected=="Tiến Trình":
        #st.subheader(f"Page: {selected}")
        Progressbar()
        graphs()

sideBar()

st.subheader('Phân Tích Theo Biểu Đồ Hộp',)
#feature_x = st.selectbox('Select feature for x Qualitative data', df_selection.select_dtypes("object").columns)
feature_y = st.selectbox('Chọn Theo Các Dữ Liệu Định Lượng', ("Điểm TBC Hệ 4", "Điểm TBC Hệ 10"))
if feature_y=="Điểm TBC Hệ 4":
    value="DiemTBCHe4"
elif feature_y=="Điểm TBC Hệ 10":
    value="DiemTBCHe10"

#sort = df_selection.sort_values(by=['XepLoaiTN'], key=lambda x: x.map(custom_dict))
# fig2 = go.Figure(
#     data=[go.Box(x=df_selection['XepLoaiTN'], y=df_selection[value])],
#     layout=go.Layout(
#         title=go.layout.Title(text="Biểu Đồ Hộp Của Điểm trung bình"),
#         plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
#         paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
#         xaxis=dict(showgrid=True, gridcolor='#cecdcd'),  # Show x-axis grid and set its color
#         yaxis=dict(showgrid=True, gridcolor='#cecdcd'),  # Show y-axis grid and set its color
#         font=dict(color='#cecdcd'),  # Set text color to black
#     )
# )

fig2 = px.box(
    df_selection.sort_values(by=['XepLoaiTN'], key=lambda x: x.map(custom_dict)),
    x="XepLoaiTN",
    y=value,
    title="Biểu đồ Hộp của Điểm trung bình theo Xếp Loại",
    labels = {"XepLoaiTN":"Xếp loại", "DiemTBCHe10":"Điểm TBC Hệ 10", "DiemTBCHe4":"Điểm TBC Hệ 4"},
    color_discrete_sequence=["#0083b8"]*len("XepLoaiTN"),
    template="plotly_white",
)

fig3 = px.box(
    df_selection,
    x="ChuyenNganh",
    y=value,
    title="Biểu đồ Hộp của Điểm trung bình theo Chuyên ngành",
    labels = {"ChuyenNganh":"Chuyên Ngành", "DiemTBCHe10":"Điểm TBC Hệ 10", "DiemTBCHe4":"Điểm TBC Hệ 4"},
    color_discrete_sequence=["#0083b8"]*len("ChuyenNganh"),
    template="plotly_white",
)

st.plotly_chart(fig2,use_container_width=True)
st.plotly_chart(fig3,use_container_width=True)

#theme
hide_st_style = """
    <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}
    </style>
"""