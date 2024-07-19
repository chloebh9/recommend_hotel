import streamlit as st
import sqlite3
import json
import requests
from PIL import Image
from io import BytesIO
from question_answer_chain import model


def show_details(user):
    with col2:
        st.markdown('<div class="result-item">', unsafe_allow_html=True)
        st.markdown(f'<h3 class="title">{user["firstName"]} {user["lastName"]}</h3>', unsafe_allow_html=True)
        st.markdown(f'<p class="label">전화번호: {user["phoneNumber"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="label">이메일: {user.get("emailAddress", "")}</p>', unsafe_allow_html=True)
        st.markdown(f'<a href="{user["homepage"]}" class="top-right-btn" target="_blank">사이트 이동</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
# SQLite 데이터베이스 연결
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Streamlit 앱 구현
st.set_page_config(layout="wide")
st.title('숙박 추천 시스템')

# CSS 스타일 정의
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://i.ibb.co/cYVPKqq/backgroundimage.jpg');
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp::before {
        position: absolute;
        content: "";
        top:0px;
        left:0px;
        width: 100%;
        height: 100%;
        background-color: rgba(255,255,255,0.7);
    }
    body {
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .stTextInput input {
        text-align: left;
        font-size: 1.5em;
    }
    .search-container {
        display: flex;
        align-items: center;
    }
    .search-container input {
        margin-right: 10px;
    }
    .result-item {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 24px;
        gap: 12px;
        margin: 10px;
        width: 800px;
        height: 208px;

        background: #1E1E1F;
        box-shadow: 0px 0px 3px rgba(18, 18, 20, 0.05), 0px 8px 16px rgba(18, 18, 20, 0.1);
        border-radius: 24px;
        position: relative;
        overflow: hidden;
    }
    .title{
        width: 600px;
        height: 44px;

        font-family: 'Inter';
        font-style: normal;
        font-weight: 800;
        font-size: 38px;
        line-height: 44px;
        /* identical to box height, or 116% */
        letter-spacing: -0.2px;

        color: #EDEDED;
    }
    .label{
        font-family: 'Inter';
        font-style: normal;
        font-weight: 700;
        font-size: 16px;
        line-height: 1px;
        color: #EDEDED;
    }

    .top-right-btn,
    .bottom-right-btn {
        position: absolute;
        background-color: #E0E0E0; /* 버튼 배경색 */
        color: #fff; /* 버튼 텍스트 색상 */
        border: none; /* 버튼 테두리 제거 */
        padding: 10px 20px; /* 버튼 내부 여백 */
        cursor: pointer; /* 마우스 커서를 포인터로 변경 */
        border-radius: 10px;
        width: 100px;
        font-size: 12px;
        color: #000000;
    }

    .top-right-btn {
        top: 10px;
        right: 10px;
    }

    .bottom-right-btn {
        bottom: 10px;
        right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 중앙 정렬을 위한 컨테이너
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_term = st.text_input('', placeholder='다음과 같이 입력해 보세요! "바다가 보이는 호텔을 추천해줘."', key='search_input')
data =''
if search_term:
    data = model(search_term)
st.markdown('</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if search_term:
    # 쓰지 못하게 된 key값
    #<p class="label">전화번호: {hotel["전화번호"]}</p>
    # <p class="label">주소: {hotel["주소"]}</p>
    # <p class="label">개요: {hotel["개요"][:24] + "...."}</p>
    # <p class="label">객실 수: {hotel["객실 수"]}</p>
    # <p class="label">부대 시설: {hotel["부대 시설"]}</p>
    # <a href="{hotel["url"]}" class="top-right-btn" target="_blank">사이트 이동</a>
        st.markdown('<div class="search-results">', unsafe_allow_html=True)
        hotel = data
        result_item = f'''
            <div class="result-item">
                <h3 class="title">{hotel["hotel_id"][0]}</h3>
                <p class="label">개요: {hotel["introduction"]}</p>
                <a class="bottom-right-btn" target="_blank">상세 정보</a>
                <button class="bottom-right-btn">상세 정보</button>
            </div>
        '''
        st.markdown(result_item, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 연결 종료
conn.close()

