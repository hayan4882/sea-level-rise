# -*- coding: utf-8 -*-
# =========================================================
# 📊 보고서 맞춤 대시보드 (민트+브라운 테마)
# ---------------------------------------------------------
# 구조:
#   1) 🥠 포춘쿠키: 버튼을 눌러 랜덤 실천 카드 보기(6개 이상)
#   2) 🔬 데이터 관측실: 더미 데이터 기반 라인/막대/지도 상호작용(지표/지역/연도 범위)
#   3) 🗂️ 자료실: 출처/참고 링크 모음 (클릭 열람)
#
# 폰트: /fonts/Pretendard-Bold.ttf (있으면 적용, 없으면 생략)
# 표준화: date, value, group
# 전처리: 결측/형변환/중복 제거/미래(로컬 자정 이후) 제거
# 캐싱: @st.cache_data
# 내보내기: 관측실의 전처리된 표 CSV 다운로드 제공
#
# ※ 공개 데이터 실제 호출은 본 앱에서는 하지 않습니다(시연용 더미 데이터).
#     공개 데이터 연결이 필요한 경우: NASA/NOAA/World Bank 등 API를 동일 스키마로 붙이면 됩니다.
#
# ★ 실행 오류 발생 시 아래 명령어로 필수 라이브러리 설치 후 실행하세요.
#     pip install -r requirements.txt
# =========================================================

import os
import random
from datetime import datetime
from typing import List
from dateutil import tz

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
from matplotlib import font_manager

APP_TITLE = "📊 내일은 물 위의 학교? — 인터랙티브 보고서 대시보드"
LOCAL_TZ = tz.gettz("Asia/Seoul")

# ---------------------------
# 폰트 적용 시도 (Pretendard)
# ---------------------------
def _try_set_pretendard() -> None:
    """시스템에 Pretendard 폰트가 있으면 적용하고, 없으면 기본 폰트 사용"""
    try:
        # Streamlit 앱 배포 환경에서는 이 경로에 폰트를 둘 수 없습니다.
        # 실제 폰트를 사용하려면 Streamlit 설정(config.toml) 또는
        # Streamlit Cloud의 Secrets를 사용해 폰트 파일을 업로드해야 합니다.
        font_path = "/fonts/Pretendard-Bold.ttf"
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            matplotlib.rcParams["font.family"] = "Pretendard"
        st.session_state.setdefault("base_font_family", "Pretendard")
    except Exception:
        # 폰트 적용 실패 시 기본 폰트로 폴백
        st.session_state.setdefault("base_font_family", "sans-serif")

_try_set_pretendard()

# ---------------------------
# 민트+브라운 테마 CSS
# ---------------------------
THEME_MINT = "#2dd4bf"   # 민트
THEME_BROWN = "#8b5e34"  # 브라운
THEME_BG = "#fbf8f5"    # 따뜻한 베이지 배경
THEME_CARD = "#ffffff"  # 카드 배경

_CSS = f"""
<style>
:root {{
  --mint: {THEME_MINT};
  --brown: {THEME_BROWN};
  --bg: {THEME_BG};
  --card: {THEME_CARD};
}}
html, body, .block-container {{ background-color: var(--bg); }}
.block-container {{padding-top: 0.8rem; padding-bottom: 1.0rem;}}
h1, h2, h3, h4 {{ color: var(--brown); margin-bottom: .5rem; }}
hr {{ margin: .6rem 0 .9rem 0; border-color: #e2e8f0; }}
.card {{
  background: var(--card);
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.03);
}}
.badge {{
  display:inline-block; padding:2px 8px; border-radius: 999px;
  background: var(--mint); color: #064e3b; font-weight:700; font-size:.75rem;
}}
.small {{ color:#6b7280; font-size:.9rem; }}
.mini {{ color:#666; font-size:.85rem; }}
.stButton>button, .stDownloadButton>button {{
  border-radius: 12px;
  border: 1px solid var(--brown);
  background: var(--mint);
  color: #064e3b;
  font-weight: 700;
}}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------
# 공통 유틸
# ---------------------------
def truncate_future_rows(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """로컬 자정 이후 데이터 제거"""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        now_local = datetime.now(LOCAL_TZ)
        today_midnight = datetime(now_local.year, now_local.month, now_local.day, tzinfo=LOCAL_TZ)
        cutoff = today_midnight.replace(tzinfo=None)
        out = out[out[date_col] < cutoff]
    return out.dropna().drop_duplicates()

def style_plot(fig: go.Figure, title: str, xlab: str, ylab: str) -> go.Figure:
    """Plotly 차트에 공통 스타일 적용"""
    base_font = st.session_state.get("base_font_family", "sans-serif")
    fig.update_layout(
        title=title,
        font=dict(family=base_font, size=14, color=THEME_BROWN),
        xaxis_title=xlab,
        yaxis_title=ylab,
        hovermode="x unified",
        margin=dict(l=40, r=18, t=50, b=36),
        paper_bgcolor=THEME_BG,
        plot_bgcolor="#ffffff",
        colorway=[THEME_MINT, THEME_BROWN, "#0ea5e9", "#65a30d"],
        legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1)
    )
    return fig

# ---------------------------
# 더미 데이터 생성기 (지표/지역별)
# ---------------------------
@st.cache_data(show_spinner=False)
def make_dummy_series(kind: str, region: str, year_start: int = 1990, year_end: int = 2025) -> pd.DataFrame:
    """지표(kind)와 지역(region)에 따라 그럴듯한 시계열 더미 생성"""
    rng = pd.date_range(f"{year_start}-01-01", f"{year_end}-12-01", freq="MS")
    base = np.linspace(0, 1, len(rng))
    noise = np.random.default_rng(42).normal(0, 0.05, len(rng))

    # 지표별 스케일/추세
    if kind == "해수면":
        trend = 30 * base   # mm 상승 가정
        seasonal = 2.0 * np.sin(np.linspace(0, 8*np.pi, len(rng)))
        values = 10 + trend + seasonal + noise*5
        unit = "mm"
    elif kind == "해수온":
        trend = 0.6 * base  # ℃ 상승 가정
        seasonal = 0.15 * np.sin(np.linspace(0, 12*np.pi, len(rng)))
        values = 0.2 + trend + seasonal + noise*0.5
        unit = "℃"
    else:   # 폭염일수
        trend = 10 * base
        seasonal = 2.5 * np.sin(np.linspace(0, 6*np.pi, len(rng)))
        values = 2 + trend + seasonal + noise*3
        values = np.clip(values, 0, None)
        unit = "일"

    # 지역 보정
    if region == "대한민국":
        values = values * 1.1 + 1.0
    else:   # 세계 평균
        values = values

    df = pd.DataFrame({
        "date": rng,
        "value": values,
        "group": f"{region}·{kind}({unit})"
    })
    df = truncate_future_rows(df, "date").sort_values("date")
    return df

@st.cache_data(show_spinner=False)
def make_dummy_bar(kind: str, region: str) -> pd.DataFrame:
    """막대용 카테고리 더미(최근 연도 기준)"""
    cats = {
        "해수면": ["침식·방파제 보강", "내륙침수 대비", "연안관리 예산", "주거이동 지원"],
        "해수온": ["어장 변화", "산호 백화", "해양열파", "연안 생태"],
        "폭염일수": ["냉방부하", "야외활동 제한", "열 관련 질환", "전력피크"]
    }
    base = np.array([40, 55, 30, 35], dtype=float)
    if region == "대한민국": base = base * 1.1
    if kind == "해수온": base = base * np.array([0.9, 1.2, 1.3, 1.0])
    if kind == "해수면": base = base * np.array([1.3, 1.1, 1.0, 1.2])
    if kind == "폭염일수": base = base * np.array([1.4, 1.2, 1.3, 1.5])
    return pd.DataFrame({"항목": cats.get(kind, []), "비율(%)": np.round(base, 1)})

def bar_percent(df: pd.DataFrame, horizontal: bool = True, title: str = "") -> go.Figure:
    """막대 차트를 생성하고 스타일을 적용"""
    if horizontal:
        fig = px.bar(df, x="비율(%)", y="항목", orientation="h", text="비율(%)")
    else:
        fig = px.bar(df, x="항목", y="비율(%)", text="비율(%)")
    fig.update_traces(textposition="outside", marker_line_color=THEME_BROWN, marker_line_width=1.2)
    return style_plot(fig, title, "비율(%)" if horizontal else "항목", "항목" if horizontal else "비율(%)")

# ----- 지도를 위한 함수를 이 부분에 추가합니다. -----
def create_interactive_map():
    """
    전 세계 해수면 상승 데이터를 시각화하는 상호작용 지도입니다.
    마우스 커서를 올리면 국가 이름과 해수면 상승률이 표시됩니다.
    """
    st.title("🌏 전세계 해수면 영향 지도")
    st.markdown(
        "마우스 커서를 올리면 **국가별 해수면 상승률**을 확인할 수 있습니다. "
        "데이터는 보고서 예시를 기반으로 합니다."
    )

    # 보고서 기반의 예시 데이터
    map_df = pd.DataFrame({
        "country": ["대한민국", "오스트레일리아", "미국", "몰디브", "방글라데시"],
        "lat": [36.5, -25.0, 37.1, 3.2, 23.7],
        "lon": [127.5, 133.0, -95.7, 73.5, 90.4],
        "sea_level_trend_mm_per_year": [3.06, 4.0, 3.3, 6.5, 5.0]
    })
    
    try:
        # Plotly를 사용하여 상호작용 지도 생성
        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="sea_level_trend_mm_per_year",
            color="sea_level_trend_mm_per_year",
            color_continuous_scale=px.colors.sequential.Teal,
            hover_name="country",
            hover_data={
                "sea_level_trend_mm_per_year": ":.2f"
            },
            title="국가별 해수면 상승률(mm/yr)",
            mapbox_style="carto-positron",
            zoom=1,
        )
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0}
        )
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"지도 시각화 중 오류가 발생했습니다: {e}")
        st.write("대신 데이터 표를 표시합니다.")
        st.dataframe(map_df)

# ---------------------------
# 청소년 피해 설문 데이터 추가
# ---------------------------
def add_youth_survey_charts():
    """청소년 피해 통계와 사례 섹션에 차트 추가"""
    st.markdown("---")
    st.subheader("청소년 대상 기후불안 설문 요약")
    
    # 데이터프레임 생성
    survey_data = {
        '항목': ['매우 그렇다', '그렇다', '불안감을 느끼지 않는다'],
        '비율': [24.8, 51.5, 23.7]
    }
    survey_df = pd.DataFrame(survey_data)

    age_data = {
        '연령대': ['만 5~12세', '만 13~18세'],
        '비율': [63.4, 36.6]
    }
    age_df = pd.DataFrame(age_data)

    # 차트 시각화
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 불안감 응답 비율")
        fig_pie = px.pie(survey_df, names='항목', values='비율', title="기후위기로 인한 불안감 응답 비율")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.markdown("### 연령대 분포")
        fig_bar = px.bar(age_df, x='연령대', y='비율', title="조사 대상 연령대 분포", labels={"연령대": "연령대", "비율": "비율(%)"})
        fig_bar.update_traces(marker_color=THEME_MINT)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("설문 데이터 다운로드")
    csv_data = survey_df.to_csv(index=False).encode('utf-8')
    st.download_button("🔽 설문 데이터 CSV 다운로드", data=csv_data, file_name="youth_survey_summary.csv", mime="text/csv")


# ---------------------------
# 상단 보고서 개요(소제목 이모티콘/클릭 요약)
# ---------------------------
def report_overview():
    """보고서 개요 섹션"""
    st.markdown("### 📌 보고서 개요")
    with st.expander("🌊 해수면 상승의 현실"):
        st.markdown("바다는 거대한 열 저장소이며, 온난화로 인해 평균 해수면이 서서히 상승하고 있어요.")
    with st.expander("🧊 뜨거워지는 지구, 녹아내리는 빙하"):
        st.markdown("해수온 상승과 빙하 융해는 해수면 상승의 두 축입니다. 바다는 열팽창으로도 높아져요.")
    with st.expander("💨 온실가스와 기후 경고"):
        st.markdown("온실가스가 많아질수록 에너지가 지구에 머물고, 극한기상 빈도가 늘어납니다.")
    with st.expander("📚 청소년과 미래 세대의 위기"):
        st.markdown("폭염·침수·해충 증가 등은 학습·건강·정서에 영향을 줍니다.")
    with st.expander("🌱 우리가 만들 해답, 우리의 실천"):
        st.markdown("교실 26℃ 유지, 칼환기, 에너지 절약, 데이터 기반 제안으로 변화를 이끌 수 있어요.")

# ---------------------------
# (1) 포춘쿠키 탭
# ---------------------------
FORTUNES: List[str] = [
    "교실 1°C 낮추기: 오후 블라인드 내리기",
    "쉬는 시간 2분 칼환기: 앞·뒤 창문 활짝!",
    "빈 교실 전원 OFF: 프로젝터·모니터 확인",
    "냉방은 26°C, 선풍기와 병행",
    "물병 챙기기: 열 스트레스 줄이기",
    "그늘길 동선 짜기: 햇빛 강한 시간 피하기",
    "우리 반 에너지 지킴이 지정하기",
    "기후 데이터 한 장 공유: 오늘의 한 그래프",
    "옥상 차열 페인트 제안서 데이터 붙이기",
    "운동장 그늘막 설치 서명받기",
]

def fortune_cookie_tab():
    """포춘쿠키 탭 내용"""
    st.markdown("### 🥠 포춘쿠키 — 오늘의 실천 한 가지")
    st.markdown('<div class="small">버튼을 눌러 오늘 바로 할 수 있는 짧고 구체적인 실천을 받아보세요.</div>', unsafe_allow_html=True)

    if "fortune" not in st.session_state:
        st.session_state["fortune"] = random.choice(FORTUNES)

    if st.button("포춘쿠키 열기 🍪"):
        st.session_state["fortune"] = random.choice(FORTUNES)

    st.markdown(
        f"""
        <div class="card">
          <span class="badge">오늘의 실천</span>
          <h4 style="margin:.4rem 0 0 0; color:var(--brown)">{st.session_state['fortune']}</h4>
          <p class="mini" style="margin:.4rem 0 0 0;">작은 행동이 모이면 교실이 달라집니다. 💚🤎</p>
        </div>
        """, unsafe_allow_html=True
    )

# ---------------------------
# (2) 데이터 관측실 탭
# ---------------------------
def data_lab_tab():
    """데이터 관측실 탭 내용"""
    st.markdown("### 🔬 데이터 관측실 — 지표·지역·기간을 바꿔보세요")
    st.markdown('<div class="small">※ 시연용 더미 데이터입니다. 실제 연결 시 NOAA/NASA/정부 공개 데이터로 교체하세요.</div>', unsafe_allow_html=True)

    colc1, colc2, colc3 = st.columns([1, 1, 2])
    with colc1:
        kind = st.selectbox("지표 선택", ["해수면", "해수온", "폭염일수"], index=0)
    with colc2:
        region = st.selectbox("지역 선택", ["대한민국", "세계 평균"], index=0)
    with colc3:
        yr = st.slider("표시 연도 범위", min_value=1980, max_value=2025, value=(1995, 2025))

    # 시계열 생성/필터
    df = make_dummy_series(kind, region, 1980, 2025)
    df = df[(df["date"].dt.year >= yr[0]) & (df["date"].dt.year <= yr[1])].copy()

    # 라인(이동평균 옵션)
    win = st.slider("스무딩(이동평균, 개월)", 1, 24, 12)
    df["MA"] = df.sort_values("date")["value"].rolling(win, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], name="월별", opacity=0.35, line=dict(color=THEME_BROWN)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA"], name=f"{win}개월 평균", line=dict(width=3, color=THEME_MINT)))
    st.plotly_chart(style_plot(fig, f"{df['group'].iloc[0]} — 시계열", "날짜", "값"), use_container_width=True)

    # 한 줄 요약 캡션
    if not df.empty:
        v0, v1 = df["MA"].iloc[0], df["MA"].iloc[-1]
        diff = v1 - v0
        arrow = "⬆️" if diff > 0 else ("⬇️" if diff < 0 else "➡️")
        st.caption(f"요약: {yr[0]}–{yr[1]} 기간 동안 {df['group'].iloc[0]}은(는) {arrow} {diff:+.2f} 변화했습니다.")

    # 막대(카테고리 영향)
    bar_df = make_dummy_bar(kind, region)
    st.plotly_chart(bar_percent(bar_df, horizontal=True, title=f"{region} {kind} 관련 영향도(시연값, %)"), use_container_width=True)

    # CSV 다운로드(전처리된 관측 데이터)
    st.download_button(
        "CSV 다운로드(관측실 데이터)",
        data=df[["date", "value", "group", "MA"]].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"observatory_{kind}_{region}_{yr[0]}_{yr[1]}.csv",
        mime="text/csv"
    )

# ---------------------------
# (3) 자료실 탭
# ---------------------------
SOURCES: List[tuple] = [
    ("데이터 가이드", "NOAA(미 해양대기청) 포털", "https://www.noaa.gov/"),
    ("데이터 가이드", "NASA Climate", "https://climate.nasa.gov/"),
    ("정부 보고", "해양수산부: 우리나라 해수면 상승 현황", "https://www.mof.go.kr/doc/ko/selectDoc.do?docSeq=44140"),
    ("참고 아틀라스", "National Atlas(해수면 영향)", "http://nationalatlas.ngii.go.kr/pages/page_3813.php"),
    ("뉴스 사례", "연합뉴스: 베트남 농작물 피해", "http://yna.co.kr/view/AKR20240318090400076"),
    ("실천 아이디어", "기후행동연구소/청소년 실천 기사", "https://climateaction.re.kr/news01/180492"),
]

def library_tab():
    """자료실 탭 내용"""
    st.markdown("### 🗂️ 자료실 — 출처/참고 링크 모음")
    st.markdown('<div class="small">클릭하면 새 창으로 열립니다. 수업·보고서에 활용하세요.</div>', unsafe_allow_html=True)
    for kind, title, url in SOURCES:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:8px;">
              <span class="badge">{kind}</span>
              <div style="font-size:18px; font-weight:700; margin-top:6px;">🔗 <a href="{url}" target="_blank">{title}</a></div>
              <div class="mini">URL: {url}</div>
            </div>
            """, unsafe_allow_html=True
        )

# ---------------------------
# 상단 보고서 본문(서론/본론/결론 요약)
# ---------------------------
def report_body():
    """보고서 본문 섹션"""
    st.markdown("## 🧭 문제 제기(서론)")
    st.markdown(
        "최근 기후 이상과 함께 해수면 상승이 눈에 띄게 진행되고 있습니다. "
        "바다는 지구의 열을 저장하고 순환시키는 거대한 장치인데, 온도가 올라가면 **열팽창**과 **빙하 융해**가 겹쳐 "
        "해수면이 서서히 높아집니다. 이 변화는 해안 침식, 내륙 침수 위험 증가, 폭염·해충 증가 등으로 학생들의 일상과 "
        "학습 환경에 직접적인 영향을 주고 있습니다.")
    st.markdown("---")

    st.markdown("## 🔍 본론 1 — 데이터로 본 해수면 상승의 현실·원인")
    st.markdown(
        "**1-1. 대한민국 해수면 변화 추이와 국제 데이터 분석**\n\n"
        "최근 기후 변화로 나타나는 폭염 현상은 단순히 대기 문제만이 아니라, 바다의 변화와도 연결되어 있다. "
        "바다는 지구에서 가장 큰 열 저장소로, 온도와 수위가 변하면 지구 전체의 기후 균형이 흔들리게 된다.\n\n"
        "따라서 해수면의 상승이 실제로 어떤 양상으로 나타나고 있는지 확인하기 위해, 우리는 전 세계와 우리나라의 데이터를 각각 살펴보고 비교 분석하였다. "
        "첫 번째로, 전 세계 평균 해수면 변화를 살펴보았다. 1880년 이후 지구 평균 해수면은 꾸준히 상승해 왔으며, 특히 1990년대 이후 그 속도가 눈에 띄게 빨라졌다. "
        "이는 빙하가 녹아 바다로 유입되고, 바닷물이 열을 받아 팽창하기 때문으로 해석된다. 결국 바다가 뜨거워지고 있다는 사실을 수치로 확인할 수 있다.\n\n"
        "이어서 1993년부터 2023년까지 위성 고도계로 측정한 호주 주변 해상 해수면 상승률 자료를 보면, 이 지역에서도 뚜렷한 상승세가 나타난다. "
        "특히 남반구 해역은 해양 순환과 기후 패턴의 영향으로 상승 속도가 빠른 편인데, 이는 특정 지역이 다른 곳보다 더 큰 위험에 노출될 수 있다는 사실을 강조한다.\n\n"
        "두 번째로, 우리나라 연안의 변화를 분석했다. 해양수산부의 관측에 따르면 지난 35년간 대한민국 연안의 평균 해수면은 약 10.7cm 상승하였다. "
        "이는 세계 평균보다 빠른 속도로, 기후 변화의 영향을 우리 사회가 직접적으로 겪고 있음을 보여준다.")
    
    # 지도를 여기에 추가합니다.
    create_interactive_map()
    
    st.markdown("---")
    st.markdown("## 🧑‍🎓 본론 1-2 — 피해 통계와 사례(청소년)")
    st.markdown(
        "**1-2. 피해 통계와 사례(청소년)**\n\n"
        "저소득층 어린이·청소년 4명 중 3명은 기후위기로 인한 불안감을 느끼고 있다는 설문조사 결과가 나왔다. "
        "환경재단은 지난달 26일부터 지난 4일까지 설문조사를 실시한 결과 저소득층 어린이·청소년 76.3%가 기후위기로 인해 불안감을 느낀다고 답했다고 밝혔다. "
        "조사 대상 어린이·청소년의 연령대: 만 5-12세 63.4%/ 만 13-18세 36.6%\n"
        "‘기후위기로 인해 불안감과 무서움을 느낀 적이 있는가?’\n"
        "‘매우 그렇다’ 24.8%\n"
        "‘그렇다’ 51.5%\n"
        "‘불안감을 느끼지 않는다’ 23.7%\n\n"
        "기후재난에 직면한 취약계층 아이들이 겪는 불평등을 조금이나마 해소하고, 미래에 대한 희망을 품을 수 있도록 지원해야함을 알 수 있다. "
        "이 세 가지 자료는 해수면 상승이 단순한 자연현상이 아니라 우리들이 만든 기후위기의 결과이며, 그 영향이 우리와 같은 청소년의 생활과 안전, 그리고 마음까지도 위협할 수 있음을 보여준다. "
        "따라서 지금 우리가 어떤 대응을 하느냐가 앞으로의 미래를 결정하는 중요한 과제임을 알 수 있다. "
        "이제 이러한 데이터를 바탕으로, 해수면 상승이 청소년과 미래 세대에 어떤 영향을 주는지 더 구체적으로 살펴보고, 나아가 정책적 대응의 필요성에 대해서도 탐구해 보겠다.")
    
    add_youth_survey_charts()

    st.markdown("---")
    st.markdown("## 🌍 본론 2 — 청소년과 미래에 미치는 영향 / 정책적 대응 필요성")
    st.markdown(
        "**2-1. 차오르는 바다와 흔들리는 청소년의 미래**\n\n"
        "기온 상승에 따른 해수면 상승은 청소년들의 생활과 건강, 심리적 안정까지 위협받고 있다. "
        "청소년들은 자신들이 마지막 세대가 될 수 있다는 불안과 무력감, 우울증에 시달리고 있으며, 폭염과 전염병 증가로 건강이 악화되고 있다. "
        "농작물 생산 감소로 식량 공급이 줄면서 영양실조에 노출되는 등 다방면에서 피해가 발생하고 있다. "
        "이러한 문제의 원인은 지구 온난화에 따른 해수면 상승과 극심한 기후변화에 있으며, 청소년들의 주거환경 불안정, 정신건강 악화, 건강 위협으로 이어진다.")
    st.markdown("---")

    st.markdown("## ✅ 결론 — 고1 눈높이로 정리한 우리의 선택")
    st.markdown(
        "해수면 상승은 멀리 있는 바다 이야기처럼 보일 수 있지만, 실제로는 교실의 온도, 등하굣길의 안전, "
        "집안의 곰팡이 같은 아주 가까운 문제로 나타납니다. 그래서 우리는 **오늘 할 수 있는 일**부터 시작해야 합니다. "
        "오후 햇빛이 강할 때 블라인드를 내려 교실 온도를 낮추고, 쉬는 시간에는 2분간 칼환기를 해 더운 공기를 내보냅니다. "
        "빈 교실의 전원을 끄는 습관을 들이면 에너지도 아끼고 열도 줄일 수 있어요. 이런 작은 실천을 반 전체가 함께하면 "
        "효과는 더 커집니다.\n\n"
        "동시에 우리는 **데이터로 말하는 힘**을 키워야 합니다. 기온·해수면 그래프를 직접 그려 보고, "
        "우리 학교 상황을 조사해보세요. 숫자와 근거를 붙여 학생회나 학교, 교육청에 **그늘막 설치**나 "
        "**차열 페인트** 같은 구체적인 개선을 요구한다면, 어른들도 더 쉽게 움직일 수 있습니다. "
        "바다가 뜨거워지는 속도를 바로 멈출 수는 없지만, 우리의 교실을 더 안전하고 시원하게 만드는 일은 "
        "지금 당장 시작할 수 있습니다. **작은 변화가 모여 내일의 학교를 바꿉니다. 💚🤎**")

# ---------------------------
# 메인
# ---------------------------
def main():
    """메인 앱 실행 함수"""
    st.set_page_config(page_title="보고서 맞춤 대시보드", layout="wide")
    st.title(APP_TITLE)

    # 상단 보고서 개요/본문(클릭형 소제목)
    report_overview()
    report_body()
    st.divider()

    tabs = st.tabs(["🥠 포춘쿠키", "🔬 데이터 관측실", "🗂️ 자료실"])
    with tabs[0]:
        fortune_cookie_tab()
    with tabs[1]:
        data_lab_tab()
    with tabs[2]:
        library_tab()

    st.caption("※ 본 앱의 데이터는 시연용 더미입니다. 실제 분석 시 NOAA/NASA/정부 공개 데이터를 동일 스키마(date, value, group)로 연결하세요.")

if __name__ == "__main__":
    main()
