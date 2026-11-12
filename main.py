import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 페이지 기본 설정
st.set_page_config(page_title="배송 지연 예측 프로그램", page_icon="📦", layout="wide")
st.title("📦 이커머스 배송 지연 예측 프로그램")
st.markdown("주문 및 배송 조건 데이터를 분석하여 **배송 지연 여부**를 예측하는 머신러닝 웹 앱입니다.")

# 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    # 캐글에서 다운로드한 원본 파일명(Train.csv) 사용
    df = pd.read_csv("Train.csv")
    # 불필요한 ID 컬럼 제거
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ 같은 폴더에 'Train.csv' 파일이 없습니다. 캐글에서 받은 CSV 파일을 넣어주세요.")
    st.stop()


# ==========================================
# 1차시: 데이터 탐색 및 시각화
# ==========================================
with tab1:
    st.header("1차시: 데이터 확인 및 시각화")
    st.write(f"전체 데이터 크기: **{df.shape[0]}행 × {df.shape[1]}열**")
    
    st.subheader("1) 데이터 미리보기")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2) 배송 지연 여부 분포 (타겟 변수)")
        st.caption("0: 정시 도착(정상) / 1: 배송 지연")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df["Reached.on.Time_Y.N"].value_counts().sort_index().plot(
            kind="bar", color=["#4CAF50", "#FF5252"], ax=ax1
        )
        ax1.set_xticklabels(["정상(0)", "지연(1)"], rotation=0)
        ax1.set_ylabel("주문 건수")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("3) 배송 수단별 주문 건수")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        df["Mode_of_Shipment"].value_counts().plot(
            kind="bar", color="#2196F3", ax=ax2
        )
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.set_ylabel("주문 건수")
        st.pyplot(fig2)
