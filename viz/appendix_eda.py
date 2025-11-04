# ============================ viz/appendix_eda.py (CLEAN — no unitprice/log) ============================
from __future__ import annotations

import numpy as np
import pandas as pd
from shiny import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= Color palette (CSS-aligned, cohesive) =================

def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

_PALETTE = {
    # UI / Typography
    "bg":     "#f6f8fb",
    "panel":  "#ffffff",
    "font":   "#1e293b",
    "muted":  "#64748b",
    "grid":   "#e2e8f0",
    "accent": "#2563eb",

    # Series / semantic
    "line":   "#123b78",
    "avg":    "#3b82f6",
    "warn":   "#f59e0b",
    "danger": "#ef4444",
    "weekday":"#94a3b8",

    # Cost / kWh (cross-page consistency)
    "cost_cur":  "#2563eb",
    "cost_prev": _rgba("#2563eb", 0.35),
    "kwh_cur":   "#10b981",
    "kwh_prev":  _rgba("#10b981", 0.35),
}
_PALETTE.setdefault("primary", _PALETTE["accent"])  # backward compat

# ==========================================================
# Layout helper (ALL-WHITE backgrounds)
# ==========================================================

def _apply_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Noto Sans KR, Inter, Arial, system-ui, sans-serif",
            size=12,
            color=_PALETTE.get("font"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=48, r=24, t=60, b=48),
    )
    fig.update_xaxes(showgrid=True, gridcolor=_PALETTE.get("grid"), zeroline=False, ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor=_PALETTE.get("grid"), zeroline=False, ticks="outside")
    return fig

# ==========================================================
# Utilities
# ==========================================================

def _to_dt(s) -> pd.Series:
    if isinstance(s, (pd.DatetimeIndex, pd.Index)):
        s = pd.Series(s)
    else:
        s = pd.Series(s)
    return pd.to_datetime(s, errors="coerce")


def _safe_replace_year(dt_like, year: int) -> pd.Series:
    s = _to_dt(dt_like)
    out = []
    for ts in s:
        if pd.isna(ts):
            out.append(pd.NaT)
            continue
        m, d, h, mi, se = ts.month, ts.day, ts.hour, ts.minute, ts.second
        if m == 2 and d == 29:
            d = 28
        try:
            out.append(pd.Timestamp(year, m, d, h, mi, se))
        except Exception:
            last = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
            out.append(pd.Timestamp(year, m, last.day, h, mi, se))
    return pd.Series(pd.to_datetime(out))


def _weekend_flag(dt_like) -> pd.Series:
    s = _to_dt(dt_like)
    return (s.dt.dayofweek >= 5).astype(int)


def _season_of_month(m: int) -> str:
    return {
        12: "겨울", 1: "겨울", 2: "겨울",
        3: "봄", 4: "봄", 5: "봄",
        6: "여름", 7: "여름", 8: "여름",
        9: "가을", 10: "가을", 11: "가을",
    }.get(m, "unknown")

# 기준 연도 고정 선언 (요청: 기초 통계 & 데이터 품질 이후는 모두 2018 기준)
_DEF_YEAR = 2018

def _force_year_2018(dt_series: pd.Series) -> pd.Series:
    return _safe_replace_year(dt_series, _DEF_YEAR)


def _holidays_by_year(year: int) -> set:
    d = set()
    def add_range(a, b):
        for dt in pd.date_range(a, b, freq="D"):
            d.add(dt.date())

    if year == 2018:
        d.add(pd.Timestamp("2018-01-01").date())
        add_range("2018-02-15", "2018-02-17")
        d.update(map(lambda x: pd.Timestamp(x).date(), [
            "2018-03-01", "2018-05-05", "2018-05-22", "2018-06-06",
            "2018-08-15", "2018-10-03", "2018-10-09", "2018-12-25"
        ]))
        add_range("2018-09-23", "2018-09-25")
    elif year == 2019:
        d.add(pd.Timestamp("2019-01-01").date())
        add_range("2019-02-04", "2019-02-06")
        d.update(map(lambda x: pd.Timestamp(x).date(), [
            "2019-03-01", "2019-05-05", "2019-05-12", "2019-06-06",
            "2019-08-15", "2019-10-03", "2019-10-09", "2019-12-25"
        ]))
        add_range("2019-09-12", "2019-09-14")
    elif year == 2021:
        d.add(pd.Timestamp("2021-01-01").date())
        add_range("2021-02-11", "2021-02-13")
        d.update(map(lambda x: pd.Timestamp(x).date(), [
            "2021-03-01", "2021-05-05", "2021-05-19", "2021-06-06",
            "2021-08-15", "2021-10-03", "2021-10-09", "2021-12-25"
        ]))
        add_range("2021-09-20", "2021-09-22")
    elif year == 2022:
        d.add(pd.Timestamp("2022-01-01").date())
        add_range("2022-01-31", "2022-02-02")
        d.update(map(lambda x: pd.Timestamp(x).date(), [
            "2022-03-01", "2022-03-09", "2022-05-05", "2022-05-08",
            "2022-06-01", "2022-06-06", "2022-10-03", "2022-10-09",
            "2022-12-25"
        ]))
        add_range("2022-09-09", "2022-09-11")
    elif year == 2023:
        d.add(pd.Timestamp("2023-01-01").date())
        add_range("2023-01-21", "2023-01-23")
        d.update(map(lambda x: pd.Timestamp(x).date(), [
            "2023-03-01", "2023-05-05", "2023-05-27", "2023-06-06",
            "2023-08-15", "2023-10-03", "2023-10-09", "2023-12-25"
        ]))
        add_range("2023-09-28", "2023-09-30")
    return d

# ==========================================================
# 1) 데이터 품질 검증
# ==========================================================

def render_calendar_alignment_storyline(df: pd.DataFrame):
    """달력 정합성 판별 요약 (막대그래프 제거)"""
    if "측정일시" not in df.columns:
        return ui.div("측정일시 컬럼 부재", class_="billx-panel p-3")

    d = df[["측정일시"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")

    has_feb29 = ((d["측정일시"].dt.month == 2) & (d["측정일시"].dt.day == 29)).any()
    leap_status = "윤년 2/29 관측됨" if has_feb29 else "현재 데이터에서 윤년 2/29 미관측"

    full_dates = d["측정일시"]
    w_ref = _weekend_flag(full_dates).to_numpy()
    candidates = [2018, 2019, 2021, 2022, 2023]
    results = []
    for yr in candidates:
        ts_y = _safe_replace_year(full_dates, yr)
        w_y = _weekend_flag(ts_y).to_numpy()
        mismatch = int((w_ref != w_y).sum())
        hols = _holidays_by_year(yr)
        hits = int(pd.Series(pd.to_datetime(ts_y)).dt.date.isin(hols).sum())
        results.append((yr, mismatch, hits, len(w_y)))

    results.sort(key=lambda x: x[1])
    best_year, best_mis, _best_hol, N = results[0]

    html = f"""
    <div class="billx-panel">
      <h6 class="billx-panel-title">1. 달력 정합성 판별</h6>
      <div class="alert alert-info mb-2">
        <strong>분석 목적:</strong> 실제 데이터의 주말/공휴일 패턴이 어느 연도 달력과 가장 일치하는지 확인
      </div>
      <ol class="mb-3">
        <li>윤년 체크: <strong>{leap_status}</strong></li>
        <li>비윤년 후보 5개년(2018, 2019, 2021, 2022, 2023)과 주말 플래그 비교</li>
      </ol>
      <div class="small text-muted mt-2">※ 공휴일은 대체/임시 공휴일 제외한 법정공휴일 기준</div>
    </div>
    """
    return ui.HTML(html)


def render_calendar_overlay(
    df: pd.DataFrame,
    year: int = 2018,
    highlight_weekend: bool = True,
    highlight_holiday: bool = True,
):
    """일별 전기요금 추이 + 주말/공휴일 하이라이트 (선택 연도 기준)"""
    if "측정일시" not in df.columns or "전기요금(원)" not in df.columns:
        return ui.div("필수 컬럼 부족", class_="billx-panel p-3")

    defaults = {
        "danger": "#ef4444", "warn": "#f59e0b", "muted": "#64748b",
        "line": "#123b78", "avg": "#3b82f6",
        "bg": "#f6f8fb", "panel": "#ffffff", "font": "#1e293b", "grid": "#e2e8f0",
    }
    pal = {**defaults, **_PALETTE}

    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")

    d["date"] = d["측정일시"].dt.normalize()
    daily = d.groupby("date", as_index=False)["전기요금(원)"].sum()

    mapped = _safe_replace_year(pd.to_datetime(daily["date"]), year)
    flags = pd.DataFrame({
        "is_weekend": _weekend_flag(mapped).astype(bool),
        "is_holiday": pd.Series(mapped).dt.date.isin(_holidays_by_year(year)),
    })

    def label_row(i: int):
        h = bool(flags.loc[i, "is_holiday"]) if highlight_holiday else False
        w = bool(flags.loc[i, "is_weekend"]) if highlight_weekend else False
        if h:
            return "공휴일"
        if w:
            return "주말"
        return "평일"

    labels = [label_row(i) for i in range(len(daily))]

    fig = go.Figure()
    fig.add_scatter(
        x=pd.to_datetime(daily["date"]),
        y=daily["전기요금(원)"],
        mode="lines",
        name="일별 합계",
        line=dict(width=2, color=pal.get("line")),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>전기요금: %{y:,.0f}원<extra></extra>",
    )

    for key, color in [("공휴일", pal.get("danger")), ("주말", pal.get("warn")), ("평일", pal.get("weekday"))]:
        idx = [i for i, v in enumerate(labels) if v == key]
        if not idx:
            continue
        fig.add_scatter(
            x=pd.to_datetime(daily["date"].iloc[idx]),
            y=daily["전기요금(원)"].iloc[idx],
            mode="markers",
            name=key,
            marker=dict(color=color, size=10, line=dict(color=pal.get("panel", "#ffffff"), width=1.2), opacity=0.95),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{meta}: %{y:,.0f}원<extra></extra>",
            meta=key,
        )

    _apply_layout(fig, title=f"일별 전기요금 추이 ({year}년 달력 기준)", height=480)
    fig.update_xaxes(title_text="날짜")
    fig.update_yaxes(title_text="전기요금(원)")

    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def render_midnight_rollover_fix(df: pd.DataFrame):
    """자정(00:00) 롤오버 검증 — 상위 5건, 원본/보정후만, 분까지 표시 (표시는 2018년 기준)"""
    if "측정일시" not in df.columns:
        return ui.div("측정일시 컬럼 부재", class_="billx-panel p-3")

    d = df[["측정일시"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")

    # 00:00 검출(검출 자체는 원시 타임스탬프 기준)
    mask = (d["측정일시"].dt.hour == 0) & (d["측정일시"].dt.minute == 0)
    midnight_data = d[mask].copy()

    n_total = len(d)
    n_midnight = len(midnight_data)
    pct = round(n_midnight / max(n_total, 1) * 100, 2)

    if n_midnight == 0:
        return ui.div(
            ui.h6("2. 자정 롤오버 검증", class_="billx-panel-title"),
            ui.div("00:00 시각 데이터가 없습니다.", class_="alert alert-success"),
            class_="billx-panel p-3",
        )

    # 화면 표시는 2018년 달력으로 강제 매핑
    orig_2018 = _force_year_2018(midnight_data["측정일시"])
    adj_2018  = _force_year_2018(midnight_data["측정일시"] + pd.Timedelta(days=1))

    sample = pd.DataFrame({
        "원본":   orig_2018.dt.strftime("%Y-%m-%d %H:%M"),
        "보정후": adj_2018.dt.strftime("%Y-%m-%d %H:%M"),
    }).head(5)
    sample_html = sample.to_html(classes="table table-sm table-bordered", index=False, border=0)

    html = f"""
    <div class="billx-panel">
      <h6 class="billx-panel-title">2. 자정(00:00) 롤오버 검증 (표시 기준: 2018년)</h6>
      <div class="alert alert-warning mb-3">
        <strong>검출:</strong> 00:00 시각 데이터 <strong>{n_midnight:,}건</strong> 발견 (전체의 {pct}%)
      </div>
      <ul class="mb-3">
        <li>00:00은 전날의 다음 시각이 아닌, <strong>다음날 00:00</strong>으로 기록된 것으로 추정</li>
        <li>날짜 경계 정합성을 위해 <code>+1일</code> 보정 필요</li>
        <li>아래는 00:00 데이터 샘플 (상위 5건, <em>원본→보정후</em>) — <strong>2018년 달력 기준으로 표기</strong></li>
      </ul>
      <div style="max-height:260px; overflow-y:auto;">{sample_html}</div>
      <div class="small text-muted mt-2">※ 검출 로직은 원시 데이터 기준, 표시는 달력 정합성 설명을 위해 2018년 기준으로 변환</div>
    </div>
    """
    return ui.HTML(html)


# ==========================================================
# 2) 기초 통계 & 데이터 품질 (요약 배지 추가)
# ==========================================================

def render_basic_stats(df: pd.DataFrame):
    """기초 통계량 + 상단 한 줄 요약 (기간 표시는 2018년 기준)"""
    n_rows, n_cols = len(df), df.shape[1]

    # 기간 표시는 2018년으로 강제 매핑해 요약(캘린더 정합성 스토리와 일관)
    date_span = "-"
    if "측정일시" in df.columns:
        dt_raw = _to_dt(df["측정일시"]).dropna().sort_values()
        if not dt_raw.empty:
            dt2018 = _force_year_2018(dt_raw)
            date_span = f"{dt2018.iloc[0].strftime('%Y-%m-%d')} ~ {dt2018.iloc[-1].strftime('%Y-%m-%d')}"

    miss_pct_max = round(df.isna().mean().max() * 100, 2) if n_rows else 0.0
    summary = f"행 {n_rows:,} / 열 {n_cols:,} · (표시 기준) 기간 {date_span} · 컬럼 최대 결측률 {miss_pct_max}%"

    # 수치형 통계 테이블(값 자체는 원시 데이터 기반)
    num = df.select_dtypes(include=[np.number])
    stats = num.describe().T
    stats["결측수"] = num.isnull().sum()
    stats["결측률(%)"] = (num.isnull().sum() / len(num) * 100).round(2)
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "결측수", "결측률(%)"]].round(2)
    stats.columns = ["개수", "평균", "표준편차", "최소", "25%", "중앙값", "75%", "최대", "결측수", "결측률(%)"]
    html_tbl = stats.to_html(classes="table table-sm table-striped", border=0)

    return ui.HTML(
        f"""
        <div class="alert alert-secondary py-2 px-3 mb-2">
          <strong>요약:</strong> {summary}
          <div class="small text-muted mt-1">※ 기간 표기는 달력 정합성 설명을 위해 2018년 기준으로 변환하였으며, 통계값 계산은 원시 데이터 기준입니다.</div>
        </div>
        <div style="max-height:420px; overflow-y:auto;">{html_tbl}</div>
        """
    )


def render_missing_summary(df: pd.DataFrame):
    m = pd.DataFrame({
        "컬럼": df.columns,
        "결측수": df.isnull().sum(),
        "결측률(%)": (df.isnull().sum() / len(df) * 100).round(2),
    })
    m = m[m["결측수"] > 0].sort_values("결측수", ascending=False)
    if len(m) == 0:
        return ui.div(ui.tags.h6("✅ 결측치 없음", class_="text-success text-center"), class_="p-3")
    html = m.to_html(classes="table table-sm table-striped", index=False, border=0)
    return ui.HTML(html)


def render_outlier_summary(df: pd.DataFrame):
    """이상치 처리 요약 — 요청: 타겟기반 항목 취소선 처리"""
    html = (
        "<div class=\"alert alert-info\">"
        "<h6 class=\"mb-2\">📋 적용된 이상치 처리</h6>"
        "<ul class=\"mb-0\">"
        "<li><del><strong>타겟 기반:</strong> 전기요금 상위 0.7% 제거</del></li>"
        "<li><strong>특정 시점:</strong> 2018-11-07 00:00:00 제거 (달력 정합성 이슈)</li>"
        "</ul>"
        "</div>"
    )

    num_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for c in num_cols:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        cnt = int(((df[c] < lo) | (df[c] > hi)).sum())
        if cnt:
            rows.append({
                "컬럼": c,
                "이상치수": cnt,
                "비율(%)": round(cnt / len(df) * 100, 2),
                "하한": round(lo, 2),
                "상한": round(hi, 2),
            })

    if rows:
        outlier_df = pd.DataFrame(rows)
        html += '<h6 class="mt-3 mb-2">📊 IQR 기준 이상치 분포</h6>'
        html += outlier_df.to_html(classes='table table-sm table-striped', index=False, border=0)
    else:
        html += '<p class="text-success mt-3">✅ IQR 기준 이상치 없음</p>'

    return ui.HTML(html)

# ==========================================================
# 3) 시계열 패턴 분석 (2018 기준)
# ==========================================================

def render_eda_storyline_panels(df: pd.DataFrame):
    if "측정일시" not in df.columns or "전기요금(원)" not in df.columns:
        return ui.div("필수 컬럼 부족", class_="billx-panel p-3")

    d = df.copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")

    dt2018 = _force_year_2018(d["측정일시"])  # 요일/일자 일관성 확보

    # 월별
    d["month"] = dt2018.dt.month
    monthly = d.groupby("month")["전기요금(원)"].sum().reset_index()
    fig_m = go.Figure()
    fig_m.add_bar(
        x=monthly["month"],
        y=monthly["전기요금(원)"],
        text=monthly["전기요금(원)"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
        marker_color=_PALETTE["cost_cur"],
    )
    _apply_layout(fig_m, title="월별 전기요금 합계 (2018년 기준)", height=360)
    fig_m.update_xaxes(title_text="월")
    fig_m.update_yaxes(title_text="전기요금(원)")

    # 일별
    d["date"] = dt2018.dt.date
    daily = d.groupby("date")["전기요금(원)"].sum().reset_index()
    fig_d = go.Figure()
    fig_d.add_scatter(
        x=daily["date"],
        y=daily["전기요금(원)"],
        mode="lines",
        line=dict(color=_PALETTE["line"], width=2),
        name="일별 합계",
    )
    _apply_layout(fig_d, title="일별 전기요금 추이 (2018년 기준)", height=360)
    fig_d.update_xaxes(title_text="날짜")
    fig_d.update_yaxes(title_text="전기요금(원)")

    # 시간별 (평균)
    d["hour"] = d["측정일시"].dt.hour
    hourly = d.groupby("hour")["전기요금(원)"].mean().reset_index()
    fig_h = go.Figure()
    fig_h.add_scatter(
        x=hourly["hour"],
        y=hourly["전기요금(원)"],
        mode="lines+markers",
        line=dict(color=_PALETTE["warn"], width=2),
        marker=dict(size=8),
        name="평균",
    )
    _apply_layout(fig_h, title="시간대별 평균 전기요금 (2018년 기준)", height=360)
    fig_h.update_xaxes(title_text="시간")
    fig_h.update_yaxes(title_text="평균 전기요금(원)")

    # 계절별 (평균)
    d["season"] = dt2018.dt.month.map(_season_of_month)
    seasonal = d.groupby("season")["전기요금(원)"].mean()
    seasonal = seasonal.reindex(["봄", "여름", "가을", "겨울"]).reset_index()
    fig_s = go.Figure()
    fig_s.add_bar(
        x=seasonal["season"],
        y=seasonal["전기요금(원)"],
        text=seasonal["전기요금(원)"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
        marker_color=_PALETTE["danger"],
    )
    _apply_layout(fig_s, title="계절별 평균 전기요금 (2018년 기준)", height=360)
    fig_s.update_xaxes(title_text="계절")
    fig_s.update_yaxes(title_text="평균 전기요금(원)")

    return ui.div(
        ui.div(
            ui.h5("시계열 패턴 분석", class_="billx-panel-title"),
            ui.div(
                "월별/일별/시간대별/계절별 전기요금 패턴으로 시간 기반 피처 설계 근거 확인",
                class_="alert alert-info mb-3",
            ),
            class_="billx-panel",
        ),
        ui.layout_columns(
            ui.div(ui.HTML(fig_m.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            ui.div(ui.HTML(fig_d.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.div(ui.HTML(fig_h.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            ui.div(ui.HTML(fig_s.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            col_widths=[6, 6],
        ),
    )

# ==========================================================
# 4) 변수 분석 (유사 항목 묶음 구성)
# ==========================================================

def plot_distribution(df: pd.DataFrame):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("전력사용량(kWh)", "전기요금(원)", "지상무효전력량(kVarh)", "지상역률(%)"),
    )

    if "전력사용량(kWh)" in df:
        fig.add_histogram(x=df["전력사용량(kWh)"], nbinsx=50, showlegend=False, row=1, col=1, marker_color=_PALETTE["kwh_cur"]) 
    if "전기요금(원)" in df:
        fig.add_histogram(x=df["전기요금(원)"], nbinsx=50, showlegend=False, row=1, col=2, marker_color=_PALETTE["cost_cur"]) 
    if "지상무효전력량(kVarh)" in df:
        fig.add_histogram(x=df["지상무효전력량(kVarh)"], nbinsx=50, showlegend=False, row=2, col=1, marker_color=_PALETTE["weekday"])  # neutral
    if "지상역률(%)" in df:
        fig.add_histogram(x=df["지상역률(%)"], nbinsx=50, showlegend=False, row=2, col=2, marker_color=_PALETTE["warn"]) 

    _apply_layout(fig, title="주요 변수 분포", height=520)
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = [
        "전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)",
        "지상역률(%)", "진상역률(%)", "탄소배출량(tCO2)", "전기요금(원)",
    ]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return ui.div("상관분석을 위한 수치형 변수 부족", class_="p-3 small-muted")

    corr = df[cols].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=[
                [0.0, _PALETTE["danger"]],
                [0.5, _PALETTE["panel"]],
                [1.0, _PALETTE["kwh_cur"]],
            ],
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(outlinewidth=0, tickcolor=_PALETTE["font"]),
        )
    )
    _apply_layout(fig, title="변수 간 상관관계", height=520)
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def plot_worktype_distribution(df: pd.DataFrame):
    if '작업유형' not in df.columns:
        return ui.div('작업유형 컬럼 없음', class_='p-3 small-muted')

    vc = df['작업유형'].value_counts()
    fig = go.Figure()

    if not vc.empty:
        fig.add_bar(
            x=vc.index,
            y=vc.values,
            text=vc.values,
            textposition='outside',
            marker_color=_PALETTE["warn"],
        )

    _apply_layout(fig, title='작업유형별 분포', height=420)
    fig.update_xaxes(title_text='작업유형')
    fig.update_yaxes(title_text='건수')

    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def plot_worktype_hourly_panels(df: pd.DataFrame):
    """작업유형 × 시간대 패턴(전력사용량, 전기요금) — 두 패널 한 번에 렌더링"""
    need_cols = {"측정일시", "작업유형", "전력사용량(kWh)", "전기요금(원)"}
    if not need_cols.issubset(df.columns):
        return ui.div("필수 컬럼 부족", class_="p-3 small-muted")

    d = df.copy()
    d["측정일시"] = _to_dt(d["측정일시"])  # 안전 변환
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")
    d["hour"] = d["측정일시"].dt.hour

    # 최신 카테고리명 가독성 정렬 (경부하/중간부하/최대부하 순)
    order = ["경부하", "중간부하", "최대부하"]
    if set(order).issubset(set(d["작업유형"].unique())):
        d["작업유형"] = pd.Categorical(d["작업유형"], categories=order, ordered=True)

    # 집계
    g_kwh = d.groupby(["작업유형", "hour"])['전력사용량(kWh)'].mean().reset_index()
    g_cost = d.groupby(["작업유형", "hour"])['전기요금(원)'].mean().reset_index()

    # kWh 패널
    fig1 = go.Figure()
    for wt, sub in g_kwh.groupby("작업유형"):
        fig1.add_scatter(
            x=sub["hour"], y=sub['전력사용량(kWh)'], mode='lines+markers', name=str(wt),
            line=dict(width=2), marker=dict(size=7)
        )
    _apply_layout(fig1, title="작업유형 × 시간대 평균 전력사용량(kWh)", height=420)
    fig1.update_xaxes(title_text='시간')
    fig1.update_yaxes(title_text='kWh')

    # 요금 패널
    fig2 = go.Figure()
    for wt, sub in g_cost.groupby("작업유형"):
        fig2.add_scatter(
            x=sub["hour"], y=sub['전기요금(원)'], mode='lines+markers', name=str(wt),
            line=dict(width=2), marker=dict(size=7)
        )
    _apply_layout(fig2, title="작업유형 × 시간대 평균 전기요금(원)", height=420)
    fig2.update_xaxes(title_text='시간')
    fig2.update_yaxes(title_text='원')

    return ui.layout_columns(
        ui.div(ui.HTML(fig1.to_html(include_plotlyjs='cdn', full_html=False)), class_=''),
        ui.div(ui.HTML(fig2.to_html(include_plotlyjs='cdn', full_html=False)), class_=''),
        col_widths=[6, 6]
    )

# ==========================================================
# 5) 파생 피처 설계 근거 (요약 버전) — 품질 검증 섹션은 요청으로 축약
# ==========================================================

def render_lag_window_acf(df: pd.DataFrame):
    if "전력사용량(kWh)" not in df.columns or "측정일시" not in df.columns:
        return ui.div("필수 컬럼 부족", class_="p-3 small-muted")

    d = df.sort_values("측정일시").copy()
    s = pd.to_numeric(d["전력사용량(kWh)"], errors="coerce")

    lags = [4, 24, 96, 192, 672, 1344]
    lag_labels = ["1h", "6h", "24h", "48h", "7d", "14d"]
    acf_vals = []
    for L, label in zip(lags, lag_labels):
        if L < len(s):
            acf_vals.append({"lag": label, "acf": float(s.autocorr(L))})

    acf_df = pd.DataFrame(acf_vals)

    fig = go.Figure()
    if not acf_df.empty:
        fig.add_bar(
            x=acf_df["lag"],
            y=acf_df["acf"],
            marker_color=_PALETTE["kwh_cur"],
            text=acf_df["acf"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
        )

    _apply_layout(fig, title="시차 상관관계 분석 (ACF)", height=360)
    fig.update_xaxes(title_text="시차")
    fig.update_yaxes(title_text="자기상관계수")

    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def render_holiday_peak_checks(df: pd.DataFrame):
    if "측정일시" not in df.columns or "전력사용량(kWh)" not in df.columns:
        return ui.div("필수 컬럼 부족", class_="p-3 small-muted")

    d = df.copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d["hour"] = d["측정일시"].dt.hour

    d["오후피크(13-17시)"] = ((d["hour"] >= 13) & (d["hour"] <= 17)).astype(int)
    d["저녁피크(18-22시)"] = ((d["hour"] >= 18) & (d["hour"] <= 22)).astype(int)
    d["심야(23-05시)"] = ((d["hour"] >= 23) | (d["hour"] <= 5)).astype(int)

    stats = []
    for flag in ["오후피크(13-17시)", "저녁피크(18-22시)", "심야(23-05시)"]:
        on_mean = d.loc[d[flag] == 1, "전력사용량(kWh)"].mean()
        off_mean = d.loc[d[flag] == 0, "전력사용량(kWh)"].mean()
        lift = (on_mean - off_mean) / (off_mean + 1e-6) * 100
        stats.append({"시간대": flag, "해당시간 평균(kWh)": round(on_mean, 2), "기타시간 평균(kWh)": round(off_mean, 2), "차이(%)": round(lift, 2)})

    stats_df = pd.DataFrame(stats)
    html = stats_df.to_html(index=False, classes="table table-sm table-striped", border=0)

    return ui.HTML(
        f"""
        <div class="billx-panel">
          <h6 class="billx-panel-title">피크시간대 플래그 유효성</h6>
          <div class="mb-2 small text-muted">피크시간대 플래그가 실제 전력 사용량에 미치는 영향을 검증.</div>
          {html}
        </div>
        """
    )
