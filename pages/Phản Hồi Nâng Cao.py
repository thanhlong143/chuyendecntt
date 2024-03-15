import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from streamlit_extras.metric_cards import style_metric_cards
from datetime import datetime

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Phân Tích Học Tập NEU", page_icon="📈", layout="wide")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

current_datetime = datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d')
formatted_day = current_datetime.strftime('%A')

st.header(" MACHINE LEARNING | MYSQL  ")
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <hr>
    <div class="card mb-3">
        <div class="card">
            <div class="card-body">
                <h3 class="card-title"style="color:#007710;"><strong> PHÂN TÍCH HỒI QUY ĐA BIẾN </strong></h3>
                <p class="card-text">Có ba biến, Điểm TBC, Điểm rèn luyện và Số tín chỉ. Mục đích là để kiểm tra xem mối quan hệ tuyến tính giữa các biến này xa đến mức nào, trong đó Điểm rèn luyện và Số tín chỉ là các đặc điểm X và Điểm TBC là đặc điểm Y. Đây là một vấn đề phân loại sử dụng phân tích hồi quy bội xác suất cho dữ liệu tồn tại trong mysql. Đo lường trực quan về các biến thể và đường phù hợp nhất</p>
                <p class="card-text"><small class="text-body-secondary"> </small></p>
            </div>
        </div>
    </div>
    <style>
        [data-testid=stSidebar] {
            color: white;
            text-size:24px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#1. read data from mysql
#2. result = view_all_data()
#3. df = pd.DataFrame(result,columns=["id","year","month","interest_rate","unemployment_rate","index_price"])

df=pd.read_excel("./datasets/Danh_sach_du_kien_HBKKHT_HK II-2022-2023.xlsx")

with st.sidebar:
    st.markdown(f"<h4 class='text-success'>{formatted_day}: {formatted_date}</h4>Analytics Dashboard V: 01/2023<hr>", unsafe_allow_html=True)

khoa= st.sidebar.multiselect(
    "CHỌN KHÓA:",
    options=df["Khoa"].unique(),
    default=df["Khoa"].unique()
)

suatHB = st.sidebar.multiselect(
    "CHỌN Suất HB HỌC KỲ II:",
    options=df["SuatHB"].unique(),
    default=df["SuatHB"].unique(),
)

df_selection = df.query(
    "Khoa == @khoa & SuatHB == @suatHB"
)

with st.sidebar:
    df_download = df_selection.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download DataFrame from Mysql",
        data=df_download,
        key="download_dataframe.csv",
        file_name="my_dataframe.csv"
)
    
df_selection.drop(columns=["Stt","MSV","Ho", "Ten", "Lop", "ChuyenNganh", "Khoa/Vien", "LoaiHB"],axis=1,inplace=True)

#theme_plotly = None 

with st.expander("EXPLORATORY ANALYSIS (Phân tích khám phá dữ liệu)"):
    st.write("Kiểm tra mối tương quan giữa các biến (đặc điểm) độc lập và biến phụ thuộc trước khi thực sự xây dựng và huấn luyện mô hình hồi quy. Đây là một bước quan trọng trong giai đoạn khám phá và phân tích dữ liệu ban đầu để hiểu mối quan hệ giữa các biến.")
    col_a,col_b=st.columns(2)
    with col_a:
        st.subheader("Số tín chỉ tích lũy vs Điểm TBC HKII")
        plt.figure(figsize=(4, 4))
        sns.regplot(x=df_selection['SoTCTichLuyHocKyII'], y=df_selection['DiemTBCHT_HKII'],color="#007710")
        plt.xlabel('Số tín chỉ tích lũy')
        plt.ylabel('Điểm TBC HKII')
        plt.title('Điểm TBC HKII vs Số tín chỉ tích lũy: Regression Plot')
        st.pyplot()

with col_b:
    plt.figure(figsize=(4, 4))
    st.subheader("Điểm rèn luyện HKII vs Điểm TBC HKII")
    sns.regplot(x=df_selection['DiemRL_HKII'], y=df_selection['DiemTBCHT_HKII'],color="#007710")
    plt.xlabel('Điểm rèn luyện HKII')
    plt.ylabel('Điểm TBC HKII')
    plt.title('Điểm rèn luyện HKII vs Điểm TBC HKII Regression Plot')
    st.pyplot()

    fig, ax = plt.subplots()
    st.subheader("Các biến ngoại lệ",)
    sns.boxplot(data=df, orient='h',color="#FF4B4B")
    plt.show()
    st.pyplot()

with st.expander("Phân tích theo tần số: HISTOGRAM"):
    df_selection.hist(figsize=(16,8),color='#007710', zorder=2, rwidth=0.9,legend = ['unemployment_rate']);
    st.pyplot()

with st.expander("Phân tích khám phá dữ liệu"):
    st.subheader("Tương quan giữa các biến",)
    #https://seaborn.pydata.org/generated/seaborn.pairplot.html
    pairplot = sns.pairplot(df_selection,plot_kws=dict(marker="+", linewidth=1), diag_kws=dict(fill=True))
    st.pyplot(pairplot.figure)

with st.expander("CÁC GIÁ TRỊ NULL, XU HƯỚNG & PHÂN TÍCH BIẾN ĐỔI"):
    a1,a2=st.columns(2)
    a1.write("số lượng các giá trị bị thiếu của mỗi cột (NaN or None)")
    a1.dataframe(df_selection.isnull().sum(),use_container_width=True)
    a2.write("cái nhìn sâu sắc về xu hướng trung tâm, sự phân tán và phân phối dữ liệu.")
    a2.dataframe(df_selection.describe().T,use_container_width=True)

# train and test split
with st.expander("Sự tương quan mặc định"):
    st.dataframe(df_selection.corr())
    st.subheader("Sự tương quan",)
    st.write("hệ số tương quan điểm TBC và số lượng tín chỉ")
    plt.scatter(df_selection['DiemTBCHT_HKII'], df_selection['SoTCTichLuyHocKyII'])
    plt.ylabel("Điểm TBC")
    plt.xlabel("Số lượng tín chỉ")
    st.pyplot()

try:
    # independent and dependent features
    X=df_selection.iloc[:,:-1] #left a last column
    y=df_selection.iloc[:,-1] #take a last column

    # train test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

    with st.expander(" SỰ PHÂN PHỐI ĐỒNG ĐỀU"):
        st.subheader("Điểm chuẩn (Z-Scores)",)
        st.write("chuyển đổi dữ liệu sao cho nó có giá trị trung bình (trung bình) bằng 0 và độ lệch chuẩn là 1. Quá trình này còn được gọi là [chia tỷ lệ] hoặc [chuẩn hóa.]")
        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.fit_transform(X_test)
        st.dataframe(X_train)

    regression=LinearRegression()
    regression.fit(X_train,y_train)

    #cross validation
    validation_score=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=3)

    col1, col3,col4,col5 = st.columns(4)
    col1.metric(label="XÁC ĐỊNH ĐIỂM TRUNG BÌNH", value=np.mean(validation_score), delta=f"{ np.mean(validation_score):,.0f}")

    #prediction
    y_pred=regression.predict(X_test)

  # performance metrics
    meansquareerror=mean_squared_error(y_test,y_pred)
    meanabsluteerror=mean_absolute_error(y_test,y_pred)
    rootmeansquareerror=np.sqrt(meansquareerror)

    col3.metric(label=" MEAN SQUARED ERROR (SAI SỐ BÌNH PHƯƠNG TRUNG BÌNH)", value=np.mean(meansquareerror), delta=f"{ np.mean(meansquareerror):,.0f}")
    col4.metric(label=" MEAN ABSOLUTE ERROR (đo độ lớn trung bình của các lỗi)", value=np.mean(meanabsluteerror), delta=f"{ np.mean(meanabsluteerror):,.0f}")
    col5.metric(label=" ROOT MEAN SQUARED ERROR (căn bậc hai của mức trung bình của các sai số bình phương)", value=np.mean(rootmeansquareerror), delta=f"{ np.mean(rootmeansquareerror):,.0f}")

    with st.expander(" COEFFICIENT OF DETERMINATION (Hệ số xác định) | R2"):
        score=r2_score(y_test,y_pred)
        st.metric(label="🔷 r", value=float(score), delta=f"{ score:,.0f}")

    with st.expander(" ADJUSTED CORRERATION COEFFICIENT (Hệ số điều chỉnh tương quan) | R"):
        st.metric(label="🔷 Adjusted R", value=float((1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))), delta=f"{ ((1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))):,.0f}")
    
    with st.expander(" CORRERATION COEFFICIENT (Hệ số tương quan) | r"):
        st.write(regression.coef_)
 
    #https://seaborn.pydata.org/generated/seaborn.regplot.html
    c1,c2,c3=st.columns(3)
    with c1:
        with st.expander(" LINE OF BEST FIT (Hồi quy tuyến tính)"):
            st.write("đường hồi quy thể hiện rõ nhất mối quan hệ giữa (các) biến độc lập và biến phụ thuộc trong mô hình hồi quy tuyến tính. Đường này được xác định thông qua một quy trình toán học nhằm giảm thiểu sai số giữa các điểm dữ liệu được quan sát và các giá trị dự đoán do mô hình tạo ra.")
            plt.figure(figsize=(8, 6))
            sns.regplot(x=y_test, y=y_pred,color="#FF4B4B",line_kws=dict(color="#007710"))
            plt.xlabel('Điểm TBC')
            plt.ylabel('Số lượng tín chỉ')
            plt.title('Điểm TBC vs Số lượng tín chỉ Regression Plot')
            st.pyplot()

    with c2:
        with st.expander(" RESIDUAL (Các phần dư)"):
            st.write("phần dư: đề cập đến sự khác biệt giữa các giá trị quan sát thực tế (biến phụ thuộc, thường được ký hiệu là y) và các giá trị dự đoán được đưa ra bởi mô hình hồi quy (thường được ký hiệu là y_pred). Các phần dư này thể hiện mức độ dự đoán của mô hình sai lệch so với các điểm dữ liệu thực tế")
            residuals=y_test-y_pred
            st.dataframe(residuals)

    with c3:
        with st.expander(" MODEL PERFORMANCE | NORMAL DISTRIBUTION CURVE (HIỆU SUẤT MÔ HÌNH | ĐƯỜNG CUNG PHÂN PHỐI BÌNH THƯỜNG) "):
            st.write("sự phân bố của một biến ngẫu nhiên liên tục trong đó dữ liệu có xu hướng được phân bố đối xứng xung quanh một giá trị trung bình (trung bình). Đây là một khái niệm cơ bản trong thống kê và lý thuyết xác suất.")
            sns.displot(residuals,kind='kde',legend=True,color="#007710")
            st.pyplot()

    with st.expander(" Ordinary Least Squares Method (Phương pháp bình phương tối thiểu)"): 
        import statsmodels.api as sm
        model=sm.OLS(y_train,X_train).fit()
        st.write(model.summary())

    style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow=True)

except:
    st.error("ERROR: LƯỢNG DỮ LIỆU KHÔNG ĐỦ ĐỂ MÔ HÌNH HOẠT ĐỘNG")