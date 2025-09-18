# streamlit_app.py
# Streamlit 앱: 한국어 UI로 공식 공개 데이터 기반 해수면 상승 대시보드 + 사용자 입력(프롬프트 기반) 대시보드
# 작성자: AI (Streamlit + GitHub Codespaces-ready)
# 출처(예시, 코드 주석으로 명확히 남김):
# - NASA/JPL Global Mean Sea Level (GMSL) & altimetry resources: https://sealevel.jpl.nasa.gov/ and https://podaac.jpl.nasa.gov/dataset/NASA_SSH_GMSL_INDICATOR  (데이터 다운로드 페이지)
# - PSMSL (Permanent Service for Mean Sea Level) tide gauge 데이터: https://psmsl.org/ (개별 관측소 및 전체 데이터)
# - NOAA Sea Level Trends / Sea Level Rise Viewer data: https://coast.noaa.gov/slrdata/ and https://tidesandcurrents.noaa.gov/sltrends/
# - 대한민국(국내) 해수면 관련 오픈데이터(예시): https://www.data.go.kr/ (국립해양조사원 관련 API 및 파일, ex: 해수면 주간 전망 통계)
#
# 위 URL들은 데이터 소스 위치(참고)이며, 실제 자동 다운로드가 불가하거나 API 실패 시 예시 데이터로 자동 대체됩니다.
# (요구사항) API 실패 시 재시도 -> 실패하면 예시 데이터로 대체하며 사용자에게 한국어 안내 표시합니다.

import io
import time
from datetime import datetime, date
from typing import Tuple

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -------------------------
# 설정: 로컬 '오늘' 기준 (개발자 지침: Asia/Seoul 타임존, 현재 날짜 2025-09-18)
# 실제 환경에서는 datetime.today() 사용하되 여기서는 시스템 시간 사용
TODAY = pd.to_datetime(datetime.now().date())

# Pretendard 폰트 시도: /fonts/Pretendard-Bold.ttf (없으면 무시)
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"

# Streamlit 페이지 설정
st.set_page_config(
    page_title="해수면 상승 대시보드 / 청소년 영향 분석",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 전역 스타일(시도): Pretendard 적용 (브라우저에서 로컬 폰트가 없으면 무시)
def inject_global_css():
    css = ""
    try:
        with open(PRETENDARD_PATH, "rb"):
            css = f"""
            <style>
            @font-face {{
                font-family: 'Pretendard';
                src: url('{PRETENDARD_PATH}');
            }}
            html, body, .css-1d391kg, .stApp {{
                font-family: 'Pretendard', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
            }}
            </style>
            """
    except Exception:
        # 파일 없으면 기본 폰트 사용
        css = """
        <style>
        html, body, .stApp { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

inject_global_css()

# -------------------------
# 유틸리티: 재시도 요청 및 예시 데이터 제공
def robust_get_csv(url: str, headers=None, timeout=10, retries=2) -> Tuple[pd.DataFrame, bool]:
    """
    URL에서 CSV를 시도하여 읽음. (재시도 허용)
    반환: (DataFrame or None, success_flag)
    """
    attempt = 0
    while attempt <= retries:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            # 판다스로 읽기
            content = r.content
            df = pd.read_csv(io.BytesIO(content))
            return df, True
        except Exception as e:
            attempt += 1
            time.sleep(1)
            if attempt > retries:
                return None, False
    return None, False

# 캐시: 외부 데이터 가져오기
@st.cache_data(show_spinner=False)
def load_public_gmsl_data() -> Tuple[pd.DataFrame, bool, str]:
    """
    NASA/JPL 또는 대체 소스에서 전지구 평균 해수면(GMSL) 시계열을 시도하여 불러옴.
    실패 시 예시 데이터 반환.
    """
    # 가능한 데이터 소스(우선순위)
    urls = [
        # JPL PO.DAAC / GMSL (사용자 환경에 따라 다운로드 URL이 다를 수 있음)
        "https://podaac.jpl.nasa.gov/dataset/NASA_SSH_GMSL_INDICATOR",
        # DataHub core sea-level-rise (CSV 제공)
        "https://datahub.io/core/sea-level-rise/r/sea-level-rise.csv",
        # Kaggle mirror (읽기 불가 가능성 있음)
        "https://raw.githubusercontent.com/datasets/global-sea-level/master/data/observations.csv",
    ]
    for url in urls:
        df, ok = robust_get_csv(url)
        if ok and isinstance(df, pd.DataFrame):
            # 표준화: date, value
            # 다양한 컬럼 이름 대응
            df_cols = [c.lower() for c in df.columns]
            # try to find year/month/date and sea level value
            if "year" in df_cols and "gmsl" in df_cols:
                # custom handling
                d = df.copy()
                if "month" in df_cols:
                    d["date"] = pd.to_datetime(d["year"].astype(int).astype(str) + "-" + d["month"].astype(int).astype(str) + "-01", errors="coerce")
                else:
                    d["date"] = pd.to_datetime(d["year"].astype(int).astype(str) + "-01-01", errors="coerce")
                # pick gmsl or 'sea_level' columns heuristically
                val_col = None
                for cand in df.columns:
                    if "gmsl" in cand.lower() or "sea_level" in cand.lower() or "absolute" in cand.lower():
                        val_col = cand
                        break
                if val_col is None:
                    val_col = df.columns[-1]
                d = d[["date", val_col]].rename(columns={val_col: "value"})
                d = d.dropna(subset=["date", "value"])
                return d, True, url
            # generic: look for a 'date' like column
            date_col = None
            for c in df.columns:
                if "date" in c.lower() or "year" in c.lower():
                    date_col = c
                    break
            value_col = None
            for c in df.columns:
                if "sea" in c.lower() and ("level" in c.lower() or "gmsl" in c.lower()):
                    value_col = c
                    break
            if date_col is None:
                # try first col as date
                date_col = df.columns[0]
            if value_col is None:
                # try last col
                value_col = df.columns[-1]
            d = df[[date_col, value_col]].copy()
            d.columns = ["date", "value"]
            # try parse date
            try:
                d["date"] = pd.to_datetime(d["date"])
            except Exception:
                # if year-only
                try:
                    d["date"] = pd.to_datetime(d["date"].astype(int).astype(str) + "-01-01")
                except Exception:
                    pass
            d = d.dropna(subset=["date", "value"])
            return d, True, url
    # 모두 실패 -> 예시 데이터 생성 (연도별 GMSL 1880~2024 가상치)
    years = np.arange(1880, 2025)
    # 누적 추세 난수 기반 (예시용)
    base = (years - 1880) * 0.2  # 단순 누적 예시
    noise = np.random.normal(loc=0.0, scale=0.5, size=len(years))
    values = base + noise
    d = pd.DataFrame({"date": pd.to_datetime(years.astype(str) + "-01-01"), "value": values})
    return d, False, "예시 데이터(내장)"

@st.cache_data(show_spinner=False)
def load_psmsl_station(korean_station_name: str = "MASAN") -> Tuple[pd.DataFrame, bool, str]:
    """
    PSMSL에서 특정 관측소(예시: MASAN) 시계열 다운로드 시도.
    실패 시 예시 관측소 데이터 반환.
    """
    # PSMSL station pages provide downloads; use known station file path pattern if available
    # 예시: https://psmsl.org/data/obtaining/stations/2044.php (MASAN)
    # 하지만 안정적 CSV URL은 보장되지 않으므로 시도 후 실패하면 예시 데이터 생성
    # 시도 URL (station id may vary)
    try_urls = [
        "https://psmsl.org/data/obtaining/stations/2044.php",  # MASAN page (html)
        "https://psmsl.org/data/psmsl.csv"  # placeholder
    ]
    for url in try_urls:
        df, ok = robust_get_csv(url)
        if ok and isinstance(df, pd.DataFrame):
            # try to standardize
            cols = df.columns
            if len(cols) >= 2:
                d = df.iloc[:, :2].copy()
                d.columns = ["date", "value"]
                try:
                    d["date"] = pd.to_datetime(d["date"])
                except Exception:
                    pass
                d = d.dropna(subset=["date", "value"])
                return d, True, url
    # 실패 -> 예시: 대한민국 연안(2004-2023) 월별 가상치(또는 연간)
    rng = pd.date_range(start="2004-01-01", end="2023-12-01", freq="MS")
    # 가벼운 상승 추세 (mm 단위)
    trend = np.linspace(0, 30, len(rng))  # 총 30mm 상승 예시
    seasonal = 5 * np.sin(np.linspace(0, 4 * np.pi, len(rng)))
    noise = np.random.normal(0, 2, len(rng))
    values = trend + seasonal + noise
    d = pd.DataFrame({"date": rng, "value": values})
    return d, False, "예시 대한민국 연안 데이터(내장)"

# -------------------------
# 사용자 입력(프롬프트 기반) 데이터: 사용자가 제공한 '보고서 계획표' 텍스트를 바탕으로
# - 실제 CSV/이미지가 업로드되지 않았으므로, 규칙에 따라 '입력 섹션'의 설명만 사용해 내부 예시 CSV 생성
# (앱 실행 중 파일 업로드/입력 요구 금지)
@st.cache_data(show_spinner=False)
def load_user_input_data() -> dict:
    """
    프롬프트의 Input 섹션(보고서 계획표)에 있는 설명만 사용하여 내부 예시 데이터 생성.
    생성 데이터:
      - korea_sealevel_20y: 지난 20년(2004-2023) 연평균 해수면(mm) 및 연평균 기온(°C) 가상 데이터
      - youth_survey: 청소년 기후 불안 설문 비율(예시: 매우그렇다/그렇다/느끼지않음)
      - youth_future_jobs: 청소년 미래 직업에 대한 기후 인식 설문(범주별 가중치 예시)
      - country_compare: 국가별 해수면 상승 예상치(예시)
    """
    # 1) 대한민국 해수면 변화 추이 (연간)
    years = np.arange(2004, 2024)
    # 가상의 연평균 해수면(mm, 상대값), 온도(°C)
    # 사용자가 문서에서 '지난 20년간 기온에 따른 해수면 상승 변화' 요구 -> 생성 시 상관관계 반영
    temp = 13.0 + 0.03 * (years - 2004) + np.random.normal(0, 0.1, len(years))  # 평균기온 가벼운 상승
    sealevel = 0.0 + 2.0 * (years - 2004) + (temp - temp.mean()) * 5 + np.random.normal(0, 3, len(years))
    korea_sealevel_20y = pd.DataFrame({"date": pd.to_datetime(years.astype(str) + "-01-01"), "value_mm": sealevel, "temp_c": temp})
    # 2) 청소년 설문(비율)
    youth_survey = pd.DataFrame({
        "응답": ["매우 그렇다", "그렇다", "불안감을 느끼지 않는다"],
        "비율": [24.8, 51.5, 23.7]
    })
    # 3) 청소년 미래 직업에 대한 기후위기 인식 (예시 항목)
    youth_future_jobs = pd.DataFrame({
        "영향_정도": ["매우 영향", "약간 영향", "영향 없음", "모름"],
        "응답수": [420, 310, 150, 70]
    })
    # 4) 국가별 해수면 상승 예상치(예시 막대그래프)
    country_compare = pd.DataFrame({
        "국가": ["대한민국", "호주", "미국", "인도네시아", "몰디브"],
        "예상_상승_cm_2100": [50, 60, 45, 75, 120]
    })
    return {
        "korea_sealevel_20y": korea_sealevel_20y,
        "youth_survey": youth_survey,
        "youth_future_jobs": youth_future_jobs,
        "country_compare": country_compare
    }

# -------------------------
# 데이터 불러오기 (공개 데이터)
public_gmsl_df, public_gmsl_ok, public_gmsl_source = load_public_gmsl_data()
psmsl_df, psmsl_ok, psmsl_source = load_psmsl_station()

# 전처리 공통 함수
def preprocess_timeseries(df: pd.DataFrame, date_col="date", value_col="value") -> pd.DataFrame:
    d = df.copy()
    # 표준 컬럼명으로 변경
    d = d.rename(columns={date_col: "date", value_col: "value"})
    # 형변환
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    # 결측 제거
    d = d.dropna(subset=["date", "value"])
    # 중복 제거: 날짜 기준 평균
    d = d.groupby("date", as_index=False).agg({"value": "mean"})
    # 미래 데이터 제거 (오늘 이후)
    d = d.loc[d["date"] <= TODAY]
    # 정렬
    d = d.sort_values("date")
    return d

# 공용 전처리
if isinstance(public_gmsl_df, pd.DataFrame):
    public_gmsl_df = preprocess_timeseries(public_gmsl_df, date_col="date", value_col="value")
else:
    public_gmsl_df = pd.DataFrame(columns=["date", "value"])

if isinstance(psmsl_df, pd.DataFrame):
    psmsl_df = preprocess_timeseries(psmsl_df, date_col="date", value_col="value")
else:
    psmsl_df = pd.DataFrame(columns=["date", "value"])

# 사용자 입력(프롬프트) 데이터 로드
user_data = load_user_input_data()

# -------------------------
# 인터페이스: 사이드바(필터 자동 구성)
st.sidebar.title("대시보드 옵션")
data_choice = st.sidebar.radio("표시 데이터 선택", ("공개 데이터: 전지구 평균 해수면", "공개 데이터: 국내 연안 관측소(예시)", "사용자 입력(보고서 기반)"))

# 날짜 범위 필터 (자동 구성)
if data_choice == "공개 데이터: 전지구 평균 해수면":
    if not public_gmsl_df.empty:
        min_d, max_d = public_gmsl_df["date"].min(), public_gmsl_df["date"].max()
    else:
        min_d, max_d = pd.to_datetime("1880-01-01"), TODAY
elif data_choice == "공개 데이터: 국내 연안 관측소(예시)":
    if not psmsl_df.empty:
        min_d, max_d = psmsl_df["date"].min(), psmsl_df["date"].max()
    else:
        min_d, max_d = pd.to_datetime("2004-01-01"), pd.to_datetime("2023-12-01")
else:
    # 사용자 데이터(대한민국 2004-2023)
    korea_df = user_data["korea_sealevel_20y"]
    min_d, max_d = korea_df["date"].min(), korea_df["date"].max()

# 사이드바: 기간 선택
start_date = st.sidebar.date_input("시작일", min_value=min_d.date(), max_value=max_d.date(), value=min_d.date())
end_date = st.sidebar.date_input("종료일", min_value=min_d.date(), max_value=max_d.date(), value=max_d.date())
if start_date > end_date:
    st.sidebar.error("시작일이 종료일보다 빠르게 설정해 주세요.")
# 스무딩 옵션 (이동평균)
smoothing = st.sidebar.slider("이동평균 윈도우(개월)", min_value=1, max_value=24, value=3)

# -------------------------
# 메인: 헤더
st.title("🌊🏫 내일은 물 위의 학교? — 해수면 상승 대시보드")
st.markdown(
    """
    **설명:** 이 대시보드는 공식 공개 데이터(NASA/PSMSL/국내 오픈데이터)를 먼저 시도하여 시각화한 뒤,
    사용자가 입력(프롬프트로 제공한 보고서 계획표)한 내용을 기반으로 별도의 대시보드를 제공합니다.
    모든 라벨과 UI는 한국어로 제공됩니다.
    """
)

# 공개 데이터 섹션
st.header("1. 공식 공개 데이터 대시보드 (자동 연결 시도)")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("전지구 평균 해수면 (GMSL) — NASA/JPL 등")
    if public_gmsl_ok:
        st.success(f"공개 데이터 로드 성공 (출처 자동 감지): {public_gmsl_source}")
    else:
        st.warning("공개 데이터 로드에 실패하여 **예시 데이터**로 대체했습니다. (네트워크 또는 소스 문제)")
        st.info("출처(참고): NASA Sea Level Change Portal / DataHub / 기타. 실제 작업 시 안정적 URL을 설정하세요.")
    # 필터 적용
    df_show = public_gmsl_df.copy()
    if not df_show.empty:
        mask = (df_show["date"] >= pd.to_datetime(start_date)) & (df_show["date"] <= pd.to_datetime(end_date))
        df_show = df_show.loc[mask]
        # 이동평균
        if smoothing > 1:
            df_show["value_smooth"] = df_show["value"].rolling(window=smoothing, min_periods=1, center=False).mean()
        else:
            df_show["value_smooth"] = df_show["value"]
        # 플롯
        fig = px.line(df_show, x="date", y=["value", "value_smooth"], labels={"value": "원시값", "value_smooth": f"{smoothing}개월 이동평균", "date":"날짜"}, title="전지구 평균 해수면 시계열")
        fig.update_layout(legend_title_text="데이터")
        st.plotly_chart(fig, use_container_width=True)
        # 표 및 다운로드
        st.markdown("#### 데이터 표 (전처리된 데이터)")
        st.dataframe(df_show.rename(columns={"date":"날짜","value":"값","value_smooth":"이동평균"}).reset_index(drop=True).head(50))
        csv = df_show.to_csv(index=False).encode('utf-8-sig')
        st.download_button("전처리된 데이터 CSV 다운로드", data=csv, file_name="public_gmsl_preprocessed.csv", mime="text/csv")
    else:
        st.info("표시할 공개 GMSL 데이터가 없습니다.")

with col2:
    st.subheader("데이터 연결 정보")
    st.markdown(f"- 전지구 평균 데이터 소스: `{public_gmsl_source}`")
    st.markdown(f"- PSMSL(관측소) 소스 예시: `{psmsl_source}`")
    st.markdown("- API 실패 시: 예시 데이터로 자동 대체됩니다.")
    st.markdown("**참고(데이터 출처 예시)**:")
    st.markdown("""
    - NASA Sea Level Change Portal (시계열/시나리오): sealevel.nasa.gov  
    - PSMSL (tide gauge): psmsl.org  
    - NOAA Sea Level Rise Viewer data: coast.noaa.gov/slrdata/
    """)

# 공개 데이터: 관측소 (국내 연안 예시)
st.header("2. 공개 데이터: 국내 연안 관측(예시) — PSMSL / KHOA 연계")
st.markdown("국내 관측소 데이터(예시)는 PSMSL에서 제공되는 관측소 데이터를 시도하여 가져옵니다. 실패 시 예시 데이터로 표시됩니다.")
if psmsl_ok:
    st.success(f"관측소 데이터 로드 성공 (출처): {psmsl_source}")
else:
    st.warning("관측소 데이터 로드 실패 — 예시 대한민국 연안 데이터로 대체되었습니다.")

if not psmsl_df.empty:
    mask = (psmsl_df["date"] >= pd.to_datetime(start_date)) & (psmsl_df["date"] <= pd.to_datetime(end_date))
    psmsl_show = psmsl_df.loc[mask].copy()
    if smoothing > 1:
        psmsl_show["value_smooth"] = psmsl_show["value"].rolling(window=smoothing, min_periods=1).mean()
    else:
        psmsl_show["value_smooth"] = psmsl_show["value"]
    fig2 = px.line(psmsl_show, x="date", y=["value", "value_smooth"], labels={"value":"원시값(mm)","value_smooth":f"{smoothing}개월 이동평균","date":"날짜"}, title="국내 연안 관측소 해수면(예시) 시계열")
    st.plotly_chart(fig2, use_container_width=True)
    csv2 = psmsl_show.to_csv(index=False).encode('utf-8-sig')
    st.download_button("국내 관측소 전처리 데이터 CSV 다운로드", data=csv2, file_name="korea_psmsl_preprocessed.csv", mime="text/csv")
else:
    st.info("국내 관측소(PSMSL) 데이터가 없습니다.")

# -------------------------
# 사용자 입력 대시보드 (프롬프트 설명 기반)
st.header("3. 사용자 입력 대시보드 (보고서 계획표 기반)")
st.markdown("아래 시각화는 사용자가 제공한 **보고서 계획표(텍스트)**를 바탕으로 자동 생성한 예시 데이터에 따른 시각화입니다. 실제 CSV/이미지를 제공하면 해당 데이터로 대체해 더 정확한 분석이 가능합니다.")

# 3-1 대한민국 해수면 변화 추이 (지난 20년) — 꺾은선 + 온도 상관
st.subheader("대한민국 해수면 변화 추이 (2004–2023) — 온도와 비교")
kdf = user_data["korea_sealevel_20y"].copy()
# 선택 기간 필터
mask = (kdf["date"] >= pd.to_datetime(start_date)) & (kdf["date"] <= pd.to_datetime(end_date))
kdf = kdf.loc[mask]
if smoothing > 1:
    kdf["value_mm_smooth"] = kdf["value_mm"].rolling(window=smoothing, min_periods=1).mean()
else:
    kdf["value_mm_smooth"] = kdf["value_mm"]
# Plot: dual axis (Plotly express workaround: make secondary y via update)
fig3 = px.line(kdf, x="date", y="value_mm_smooth", labels={"value_mm_smooth":"해수면 변화 (mm)","date":"연도"}, title="대한민국 연안: 연평균 해수면 변화 (예시)")
fig3.add_scatter(x=kdf["date"], y=kdf["temp_c"], mode="lines", name="연평균기온(°C)", yaxis="y2")
fig3.update_layout(
    yaxis=dict(title="해수면 변화 (mm)"),
    yaxis2=dict(title="연평균기온 (°C)", overlaying="y", side="right"),
    legend_title_text="지표",
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("**해석(예시):** 온도가 서서히 상승함에 따라(우측 축) 해수면도 장기적으로 상승 경향을 보입니다. 청소년·지역사회 영향 분석의 기초 자료로 활용하세요.")
csv_korea = kdf.to_csv(index=False).encode('utf-8-sig')
st.download_button("대한민국 해수면(보고서 기반) CSV 다운로드", data=csv_korea, file_name="korea_sealevel_2004_2023.csv", mime="text/csv")

# 3-2 국가별 해수면 상승 예상치 — 막대그래프
st.subheader("국가별 해수면 상승 예상치 비교 (예시)")
country_df = user_data["country_compare"]
fig4 = px.bar(country_df, x="국가", y="예상_상승_cm_2100", labels={"예상_상승_cm_2100":"2100년 예상 상승 (cm)"}, title="국가별 해수면 상승 예상치 (예시)")
st.plotly_chart(fig4, use_container_width=True)
csv_country = country_df.to_csv(index=False).encode('utf-8-sig')
st.download_button("국가별 예상치 CSV 다운로드", data=csv_country, file_name="country_sea_level_projection_example.csv", mime="text/csv")

# 3-3 피해 통계와 사례(청소년) — 원그래프 및 막대그래프
st.subheader("피해 통계(청소년) — 기후 불안 응답 비율")
youth_survey = user_data["youth_survey"]
fig5 = px.pie(youth_survey, names="응답", values="비율", title="기후위기로 인한 불안감 비율(청소년, 예시)")
st.plotly_chart(fig5, use_container_width=True)
st.markdown("**설명(보고서 기반):** 저소득층 어린이·청소년의 76.3%가 기후위기 때문에 불안감을 느낀다는 조사(보고서 예시)를 반영한 비율입니다.")

st.subheader("청소년: 기후위기가 미래 직업/진로에 미치는 영향 (예시)")
yfj = user_data["youth_future_jobs"]
fig6 = px.bar(yfj, x="영향_정도", y="응답수", labels={"응답수":"응답 수", "영향_정도":"영향 정도"}, title="청소년의 기후위기 인식(미래 직업에 대한 영향, 예시)")
st.plotly_chart(fig6, use_container_width=True)
csv_youth = yfj.to_csv(index=False).encode('utf-8-sig')
st.download_button("청소년 설문(예시) CSV 다운로드", data=csv_youth, file_name="youth_survey_example.csv", mime="text/csv")

# 3-4 텍스트 요약(보고서 템플릿) — 보고서 제목 및 서론/결론 템플릿 제공
st.header("4. 보고서 작성 도움: 템플릿 (제목·서론·결론 예시)")
st.markdown("**보고서 제목(가제)**: 🌊🏫 내일은 물 위의 학교? : 🚨 해수면 상승의 경고")
st.subheader("서론 (예시)")
st.write(
    "최근 기후이상으로 폭염 및 자연재해가 증가하고 있으며, 지난 30년(1991–2020) 동안 한국 연안의 평균 해수면이 연평균 3.03mm 증가하여 약 9.1cm 상승한 연구 결과가 보고되었습니다. "
    "이에 본 보고서는 해수면 상승이 청소년 세대에 미치는 영향(심리·생활·진로)을 데이터 기반으로 분석하고 정책적 제언을 제시합니다."
)
st.subheader("결론 및 제언 (예시)")
st.write(
    "해수면 상승은 단순한 자연현상이 아니라 기후위기의 결과이며, 청소년의 정신건강·주거·진로에 장기적 영향을 미칩니다. "
    "정책 제언: (1) 학교 내 기후 심리 지원체계 구축, (2) 취약계층 대상 재난 안전망 강화, (3) 해안지역 장기 모니터링 및 이주·보호 정책 수립."
)

# -------------------------
# 추가 정보: Kaggle API 인증 안내 (요청시 자동 표출)
with st.expander("🔎 kaggle API 사용/인증 방법 (선택)"):
    st.markdown(
        """
        Kaggle에서 데이터셋을 자동으로 가져오려면 `kaggle` 패키지를 사용해야 합니다.
        1. Kaggle 계정에서 *My Account* -> *API*에서 `kaggle.json`을 발급받으세요.  
        2. Codespaces/로컬 환경에서 `~/.kaggle/kaggle.json`으로 업로드하거나 환경변수로 설정하세요.  
        3. 예) `pip install kaggle` 후 `kaggle datasets download -d <dataset-owner/dataset-name>` 사용.  
        (주의) 이 앱은 현재 네트워크 또는 접근 권한에 따라 직접 kaggle 다운로드가 실패할 수 있으며, 실패 시 예시 데이터로 대체됩니다.
        """
    )

# -------------------------
# 앱 하단: 데이터 처리 원칙 및 주의사항
st.sidebar.markdown("---")
st.sidebar.header("데이터 처리 원칙")
st.sidebar.markdown(
    """
    - 표준화된 컬럼: `date`, `value` (또는 변형 컬럼)을 우선 사용합니다.  
    - 전처리: 결측치 처리, 형 변환, 중복 제거, 미래 날짜(오늘 이후) 제거를 수행합니다.  
    - 캐싱: `@st.cache_data`로 외부 호출 캐시 적용(재실행 속도 향상).  
    - CSV 다운로드 버튼을 통해 전처리된 표를 제공합니다.
    """
)

st.caption("앱에서 제공된 일부 데이터는 네트워크/API 실패 시 내부 예시 데이터로 대체되었습니다. 실제 공개 데이터 사용 시에는 위 주석의 출처 URL을 확인하고 적절한 인증 및 파일 경로를 설정하세요.")
