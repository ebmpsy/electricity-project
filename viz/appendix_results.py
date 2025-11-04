# =====================================================================
# viz/appendix_results.py  (Tab: 결과/검증)
# - render_metrics_table
# - render_residual_plot
# - render_shap_summary
# - render_shap_bar
# - render_deploy_checklist
# =====================================================================
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from shiny import ui
import plotly.graph_objects as go
from viz.appendix_common import apply_layout, _PALETTE
from pathlib import Path
from scipy.stats import kurtosis, norm
import plotly.figure_factory as ff
from plotly.subplots import make_subplots



def _ph(text: str = "여기에 표/그래프가 표시됩니다.", h: int = 260):
    """統一 placeholder (톤앤매너 유지)"""
    return ui.div(
        text,
        class_="placeholder d-flex align-items-center justify-content-center small-muted",
        style=f"height:{h}px; font-size: 0.98rem;",
    )


# ---------------------------------------------------------------------
# 1) 평가 지표 표
# ---------------------------------------------------------------------



import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis
import plotly.figure_factory as ff
import plotly.graph_objects as go
from shiny import ui


import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import ui


# ---------------------------------------------------------------------
# 1) 모델별 잔차 분포 + 대표 모델 σ 커버리지 표시
# ---------------------------------------------------------------------
def render_shap_summary():
    """
    holdout_predictions.csv를 기반으로 모델별 잔차(Residual) 분포를 표시하고,
    대표 모델(첫 번째 모델)의 ±σ 수직선을 시각화합니다.
    """
    csv_path = Path(__file__).resolve().parents[1] / "data" / "output" / "holdout_predictions.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return ui.div("❌ holdout_predictions.csv 파일을 찾을 수 없습니다.", class_="alert alert-danger")
    except Exception as e:
        return ui.div(f"❌ 파일 로드 오류: {str(e)}", class_="alert alert-danger")

    ACTUAL_COL = "실제요금"
    if ACTUAL_COL not in df.columns:
        return ui.div(f"❌ '{ACTUAL_COL}' 컬럼을 찾을 수 없습니다.", class_="alert alert-warning")

    PRED_COLS = [c for c in df.columns if c.endswith("_pred")]
    if not PRED_COLS:
        return ui.div("❌ '_pred'로 끝나는 예측 컬럼이 없습니다.", class_="alert alert-warning")

    # 모델별 품질 메트릭 계산
    metrics = []
    for col in PRED_COLS:
        model = col.replace("_pred", "")
        residuals = df[col] - df[ACTUAL_COL]
        residuals = residuals.dropna()
        
        if len(residuals) == 0:
            continue
            
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        std = np.std(residuals)
        kurt_val = kurtosis(residuals)
        c1 = np.mean((np.abs(residuals) <= std)) * 100
        c2 = np.mean((np.abs(residuals) <= 2 * std)) * 100
        c3 = np.mean((np.abs(residuals) <= 3 * std)) * 100
        metrics.append((model, mae, rmse, std, kurt_val, c1, c2, c3))

    if not metrics:
        return ui.div("❌ 계산 가능한 메트릭이 없습니다.", class_="alert alert-warning")

    metric_df = pd.DataFrame(metrics, columns=["모델", "MAE", "RMSE", "STD", "Kurtosis", "±1σ", "±2σ", "±3σ"])

    # 잔차 분포 그래프 (KDE)
    hist_data, labels = [], []
    for col in PRED_COLS:
        residuals = (df[col] - df[ACTUAL_COL]).dropna()
        if len(residuals) > 0:
            hist_data.append(residuals.values)
            labels.append(col.replace("_pred", ""))

    if not hist_data:
        return ui.div("❌ 그래프 생성을 위한 데이터가 없습니다.", class_="alert alert-warning")

    try:
        fig_resid = ff.create_distplot(hist_data, labels, show_hist=True, show_rug=False)
        fig_resid.update_layout(
            title=dict(text="<b>모델별 잔차(Residual) 분포 및 대표 σ 경계</b>", font=dict(size=18), x=0.5),
            xaxis_title="잔차 (예측값 - 실제값)",
            yaxis_title="Density",
            plot_bgcolor="white",
            xaxis=dict(gridcolor="lightgrey"),
            yaxis=dict(gridcolor="lightgrey"),
            height=500,
            showlegend=True,
        )
    except Exception as e:
        return ui.div(f"❌ 그래프 생성 오류: {str(e)}", class_="alert alert-danger")

    # 대표 모델 기준 ±σ 수직선 추가
    first_model = PRED_COLS[0]
    first_name = first_model.replace("_pred", "")
    first_residuals = (df[first_model] - df[ACTUAL_COL]).dropna()
    mean_val = np.mean(first_residuals)
    std_val = np.std(first_residuals)

    sigma_levels = [1, 2, 3]
    colors = ["red", "orange", "gray"]

    for sigma, color in zip(sigma_levels, colors):
        fig_resid.add_vline(
            x=mean_val + sigma * std_val,
            line=dict(color=color, width=1.5, dash="dot"),
            annotation_text=f"+{sigma}σ",
            annotation_position="top right"
        )
        fig_resid.add_vline(
            x=mean_val - sigma * std_val,
            line=dict(color=color, width=1.5, dash="dot"),
            annotation_text=f"-{sigma}σ",
            annotation_position="top left"
        )

        # 품질 요약 표
        rows_html = "".join(
                """
                <tr>
                    <td class='fw-semibold'>{model}</td>
                    <td>{mae:.3f}</td>
                    <td>{std:.3f}</td>
                    <td>{kurt:.2f}</td>
                    <td>{c1:.2f}%</td>
                    <td>{c2:.2f}%</td>
                    <td>{c3:.2f}%</td>
                </tr>
                """.format(
                        model=r["모델"],
                        mae=r["MAE"],
                        std=r["STD"],
                        kurt=r["Kurtosis"],
                        c1=r["±1σ"],
                        c2=r["±2σ"],
                        c3=r["±3σ"],
                )
                for _, r in metric_df.iterrows()
        )

        desc_html = f"""
        <div class='p-3'>
            <h5>📏 품질 요약 (대표 모델: {first_name})</h5>
            <div class='table-responsive'>
                <table class='table table-sm table-striped align-middle mb-0' style='font-size:0.92rem;'>
                    <thead class='table-light'>
                        <tr>
                            <th>모델</th>
                            <th>MAE</th>
                            <th>STD</th>
                            <th>Kurtosis</th>
                            <th>±1σ</th>
                            <th>±2σ</th>
                            <th>±3σ</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>
        """

    html = (
        '<div class="billx-panel">'
        + fig_resid.to_html(include_plotlyjs="cdn", full_html=False)
        + '</div>'
        + desc_html
    )

    return ui.HTML(html)


# ---------------------------------------------------------------------
# 2) ±3σ 이상치 지상역률(%) 분포 분석
# ---------------------------------------------------------------------
def render_metrics_table():
    """
    ±3σ 이상 잔차 시점의 지상역률(%) 분포를 원본 전체와 비교 분석합니다.
    """
    base_dir = Path("./data")
    pred_path = base_dir / "output" / "holdout_predictions.csv"
    train_path = base_dir / "train.csv"

    try:
        df_pred = pd.read_csv(pred_path)
        df_train = pd.read_csv(train_path)
    except FileNotFoundError as e:
        return ui.div(f"❌ CSV 파일을 찾을 수 없습니다: {str(e)}", class_="alert alert-danger")
    except Exception as e:
        return ui.div(f"❌ 파일 로드 오류: {str(e)}", class_="alert alert-danger")

    # 필수 컬럼 확인
    if "지상역률(%)" not in df_train.columns:
        return ui.div("❌ train.csv에 '지상역률(%)' 컬럼이 없습니다.", class_="alert alert-warning")
    
    if "실제요금" not in df_pred.columns:
        return ui.div("❌ holdout_predictions.csv에 '실제요금' 컬럼이 없습니다.", class_="alert alert-warning")

    # 대표 모델 선택
    pred_cols = [c for c in df_pred.columns if c.endswith("_pred")]
    if not pred_cols:
        return ui.div("❌ '_pred'로 끝나는 예측 컬럼이 없습니다.", class_="alert alert-warning")

    target_col = pred_cols[0]
    model_name = target_col.replace("_pred", "")
    df_pred["Residual"] = df_pred[target_col] - df_pred["실제요금"]

    # 데이터 병합
    if len(df_pred) <= len(df_train):
        df_merge = df_pred.copy()
        df_merge["지상역률(%)"] = df_train["지상역률(%)"].iloc[:len(df_pred)].values
    else:
        return ui.div("❌ 예측 데이터가 원본보다 깁니다.", class_="alert alert-warning")

    # ±3σ 기준 이상치 추출
    residuals = df_merge["Residual"].dropna()
    if len(residuals) == 0:
        return ui.div("❌ 잔차 데이터가 없습니다.", class_="alert alert-warning")
        
    mean_resid = residuals.mean()
    std_resid = residuals.std()
    upper_bound = mean_resid + 3 * std_resid
    lower_bound = mean_resid - 3 * std_resid

    df_normal = df_merge[(df_merge["Residual"] >= lower_bound) & (df_merge["Residual"] <= upper_bound)]
    df_outlier = df_merge[(df_merge["Residual"] > upper_bound) | (df_merge["Residual"] < lower_bound)]

    # 지상역률(%) 분포 비교 그래프
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("전체 데이터 지상역률(%) 분포", "±3σ 이상치 지상역률(%) 분포"),
        horizontal_spacing=0.15
    )

    # 전체 지상역률(%) 분포
    pf_all = df_merge["지상역률(%)"].dropna()
    fig.add_trace(
        go.Histogram(x=pf_all, nbinsx=30, name="전체", marker_color="lightblue", showlegend=False),
        row=1, col=1
    )

    # 이상치 지상역률(%) 분포
    if len(df_outlier) > 0:
        pf_outlier = df_outlier["지상역률(%)"].dropna()
        fig.add_trace(
            go.Histogram(x=pf_outlier, nbinsx=30, name="±3σ 이상치", marker_color="red", showlegend=False),
            row=1, col=2
        )

    fig.update_xaxes(title_text="지상역률(%)", row=1, col=1)
    fig.update_xaxes(title_text="지상역률(%)", row=1, col=2)
    fig.update_yaxes(title_text="빈도", row=1, col=1)
    fig.update_yaxes(title_text="빈도", row=1, col=2)

    fig.update_layout(
        title=dict(text="<b>지상역률(%) 분포 비교: 전체 vs ±3σ 이상치</b>", x=0.5, font=dict(size=18)),
        height=450,
        plot_bgcolor="white",
        showlegend=False
    )

    # 통계 분석
    analysis_html = "<div class='p-3'><h5>📊 지상역률(%) 분포 분석</h5>"
    
    pf_all_mean = pf_all.mean()
    pf_all_std = pf_all.std()
    
    analysis_html += f"<p><b>전체 데이터:</b> 평균 지상역률(%) {pf_all_mean:.3f}, 표준편차 {pf_all_std:.3f}</p>"
    
    if len(df_outlier) > 0:
        pf_out_mean = pf_outlier.mean()
        pf_out_std = pf_outlier.std()
        diff = pf_out_mean - pf_all_mean
        
        analysis_html += f"<p><b>±3σ 이상치:</b> 평균 지상역률(%) {pf_out_mean:.3f}, 표준편차 {pf_out_std:.3f}</p>"
        analysis_html += f"<p><b>차이:</b> {abs(diff):.3f} ({'+' if diff > 0 else ''}{diff:.3f})</p>"
        
        # 해석
        if abs(diff) < 0.02:
            analysis_html += "<p style='color: green;'>✅ 지상역률(%) 차이가 매우 작습니다 (0.02 미만). 이상치는 <b>지상역률(%)과 무관</b>하게 발생한 것으로 보입니다.</p>"
        elif abs(diff) < 0.05:
            analysis_html += "<p style='color: orange;'>⚠️ 지상역률(%) 차이가 다소 있습니다 (0.02~0.05). 지상역률(%)이 이상치 발생에 <b>일부 영향</b>을 줄 수 있습니다.</p>"
        else:
            analysis_html += f"<p style='color: red;'>🚨 지상역률(%) 차이가 큽니다 (0.05 이상). ±3σ 이상치는 <b>{'높은' if diff > 0 else '낮은'} 지상역률(%)</b> 구간에서 주로 발생합니다.</p>"
            
        # 시간대/월별 분석 간단히
        if "측정일시" in df_train.columns:
            df_merge["측정일시"] = pd.to_datetime(df_train["측정일시"].iloc[:len(df_merge)], errors='coerce')
            df_merge["hour"] = df_merge["측정일시"].dt.hour
            df_merge["month"] = df_merge["측정일시"].dt.month
            df_merge["weekday"] = df_merge["측정일시"].dt.dayofweek
            
            df_outlier_time = df_merge[(df_merge["Residual"] > upper_bound) | (df_merge["Residual"] < lower_bound)]
            
            hour_dist = df_outlier_time["hour"].value_counts(normalize=True) * 100
            month_dist = df_outlier_time["month"].value_counts(normalize=True) * 100
            weekday_dist = df_outlier_time["weekday"].value_counts(normalize=True) * 100
            
            analysis_html += "<hr><h5>⏰ 시간적 패턴</h5>"
            
            # 월별
            if len(month_dist) > 0:
                max_month_pct = month_dist.max()
                min_month_pct = month_dist.min()
                if max_month_pct - min_month_pct < 5:
                    analysis_html += f"<p>• <b>월별:</b> 차이 없음 (최대 {max_month_pct:.1f}% - 최소 {min_month_pct:.1f}% = {max_month_pct - min_month_pct:.1f}%p)</p>"
                else:
                    top_month = month_dist.idxmax()
                    analysis_html += f"<p>• <b>월별:</b> {int(top_month)}월에 집중 ({month_dist[top_month]:.1f}%)</p>"
            
            # 시간대
            if len(hour_dist) > 0:
                max_hour_pct = hour_dist.max()
                min_hour_pct = hour_dist.min()
                if max_hour_pct - min_hour_pct < 5:
                    analysis_html += f"<p>• <b>시간대:</b> 차이 없음 (최대 {max_hour_pct:.1f}% - 최소 {min_hour_pct:.1f}% = {max_hour_pct - min_hour_pct:.1f}%p)</p>"
                else:
                    top_hour = hour_dist.idxmax()
                    analysis_html += f"<p>• <b>시간대:</b> {int(top_hour)}시에 집중 ({hour_dist[top_hour]:.1f}%)</p>"
            
            # 요일
            day_map = ["월", "화", "수", "목", "금", "토", "일"]
            if len(weekday_dist) > 0:
                max_day_pct = weekday_dist.max()
                min_day_pct = weekday_dist.min()
                if max_day_pct - min_day_pct < 5:
                    analysis_html += f"<p>• <b>요일:</b> 차이 없음 (최대 {max_day_pct:.1f}% - 최소 {min_day_pct:.1f}% = {max_day_pct - min_day_pct:.1f}%p)</p>"
                else:
                    top_day = weekday_dist.idxmax()
                    analysis_html += f"<p>• <b>요일:</b> {day_map[int(top_day)]}요일에 집중 ({weekday_dist[top_day]:.1f}%)</p>"
    else:
        analysis_html += "<p style='color: green;'>✅ ±3σ 이상치가 없습니다. 모델이 안정적입니다.</p>"
    
    analysis_html += "</div>"

    html = (
        '<div class="billx-panel">'
        + fig.to_html(include_plotlyjs="cdn", full_html=False)
        + '</div>'
        + analysis_html
    )

    return ui.HTML(html)


# 별칭
# ---------------------------------------------------------------------
# 2) ±3σ 이상치 진상역률(%) 분포 분석
# ---------------------------------------------------------------------
def render_residual_plot():
    """
    ±3σ 이상 잔차 시점의 진상역률(%) 분포를 원본 전체와 비교 분석합니다.
    """
    base_dir = Path("./data")
    pred_path = base_dir / "output" / "holdout_predictions.csv"
    train_path = base_dir / "train.csv"

    try:
        df_pred = pd.read_csv(pred_path)
        df_train = pd.read_csv(train_path)
    except FileNotFoundError as e:
        return ui.div(f"❌ CSV 파일을 찾을 수 없습니다: {str(e)}", class_="alert alert-danger")
    except Exception as e:
        return ui.div(f"❌ 파일 로드 오류: {str(e)}", class_="alert alert-danger")

    # 필수 컬럼 확인
    if "진상역률(%)" not in df_train.columns:
        return ui.div("❌ train.csv에 '진상역률(%)' 컬럼이 없습니다.", class_="alert alert-warning")
    if "실제요금" not in df_pred.columns:
        return ui.div("❌ holdout_predictions.csv에 '실제요금' 컬럼이 없습니다.", class_="alert alert-warning")

    # 대표 모델 선택
    pred_cols = [c for c in df_pred.columns if c.endswith("_pred")]
    if not pred_cols:
        return ui.div("❌ '_pred'로 끝나는 예측 컬럼이 없습니다.", class_="alert alert-warning")

    target_col = pred_cols[0]
    model_name = target_col.replace("_pred", "")
    df_pred["Residual"] = df_pred[target_col] - df_pred["실제요금"]

    # 데이터 병합
    if len(df_pred) <= len(df_train):
        df_merge = df_pred.copy()
        df_merge["진상역률(%)"] = df_train["진상역률(%)"].iloc[:len(df_pred)].values
    else:
        return ui.div("❌ 예측 데이터가 원본보다 깁니다.", class_="alert alert-warning")

    # ±3σ 기준 이상치 추출
    residuals = df_merge["Residual"].dropna()
    mean_resid = residuals.mean()
    std_resid = residuals.std()
    upper_bound = mean_resid + 3 * std_resid
    lower_bound = mean_resid - 3 * std_resid

    df_outlier = df_merge[(df_merge["Residual"] > upper_bound) | (df_merge["Residual"] < lower_bound)]

    # 진상역률(%) 분포 비교 그래프
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("전체 데이터 진상역률(%) 분포", "±3σ 이상치 진상역률(%) 분포"),
        horizontal_spacing=0.15
    )

    pf_all = df_merge["진상역률(%)"].dropna()
    fig.add_trace(
        go.Histogram(x=pf_all, nbinsx=30, name="전체", marker_color="lightblue", showlegend=False),
        row=1, col=1
    )

    if len(df_outlier) > 0:
        pf_outlier = df_outlier["진상역률(%)"].dropna()
        fig.add_trace(
            go.Histogram(x=pf_outlier, nbinsx=30, name="±3σ 이상치", marker_color="red", showlegend=False),
            row=1, col=2
        )

    fig.update_xaxes(title_text="진상역률(%)", row=1, col=1)
    fig.update_xaxes(title_text="진상역률(%)", row=1, col=2)
    fig.update_yaxes(title_text="빈도", row=1, col=1)
    fig.update_yaxes(title_text="빈도", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>진상역률(%) 분포 비교: 전체 vs ±3σ 이상치</b>", x=0.5, font=dict(size=18)),
        height=450,
        plot_bgcolor="white"
    )

    # 분석 텍스트
    analysis_html = "<div class='p-3'><h5>📊 진상역률(%) 이상치 분석</h5>"

    pf_all_mean = pf_all.mean()
    pf_all_std = pf_all.std()

    if len(df_outlier) > 0:
        pf_out_mean = df_outlier["진상역률(%)"].mean()
        diff = pf_out_mean - pf_all_mean

        analysis_html += f"""
        <p><b>전체 평균 진상역률(%)</b>: {pf_all_mean:.3f} |
        <b>±3σ 이상치 평균</b>: {pf_out_mean:.3f} |
        <b>차이</b>: {diff:+.3f}</p>
        """

        if abs(diff) < 0.02:
            analysis_html += "<p style='color:green;'>✅ 차이가 매우 작습니다. 진상역률(%)과 이상치는 거의 무관합니다.</p>"
        elif abs(diff) < 0.05:
            analysis_html += "<p style='color:orange;'>⚠️ 약간의 차이가 있습니다. 진상역률(%) 변화가 일부 영향을 미칠 수 있습니다.</p>"
        else:
            trend = "높은" if diff > 0 else "낮은"
            analysis_html += f"<p style='color:red;'>🚨 ±3σ 이상치는 <b>{trend} 진상역률(%)</b> 구간에서 집중 발생합니다.</p>"
    else:
        analysis_html += "<p style='color:green;'>✅ ±3σ 이상치가 없습니다. 모델의 안정성이 우수합니다.</p>"

    analysis_html += "</div>"

    html = (
        '<div class="billx-panel">'
        + fig.to_html(include_plotlyjs="cdn", full_html=False)
        + '</div>'
        + analysis_html
    )

    return ui.HTML(html)




# ---------------------------------------------------------------------
# 4) SHAP Bar (특정 샘플/집단 평균의 feature 영향 Top-K)
# ---------------------------------------------------------------------
def render_shap_bar(

):
   
    return 0

# ---------------------------------------------------------------------
# 5) 배포/모니터링 체크리스트
# ---------------------------------------------------------------------
def render_deploy_checklist():
    html = """
    <div class="p-3" style="font-size: 0.98rem;">
      <div class="alert alert-primary">
        <b>배포/모니터링 체크리스트</b>
      </div>
      <ul class="mb-3">
        <li><b>피처 일관성</b>: 학습/추론 파이프라인 동일(결측 처리·스케일·라벨링·캘린더 기준연도)</li>
        <li><b>입력 검증</b>: 스키마/범위(이상치·음수·시간 역전)/00:00 롤오버 보정 여부</li>
        <li><b>드리프트 감시</b>: 데이터/타겟/에러(예: MAPE/MAE의 주간 이동평균), 경보 임계치</li>
        <li><b>재학습 정책</b>: 주기/트리거(성능 하락·분포 변화·설비 변경 등)와 모델 버저닝</li>
        <li><b>성능 추적</b>: Holdout/Online A/B, 예측·실측 대시보드(주말/공휴일 분리)</li>
        <li><b>로깅</b>: 입력/출력/특성량/지표/추론시간, 실패 재처리 전략</li>
        <li><b>보안/권한</b>: 환경변수, 자격증명, 민감 데이터 마스킹</li>
        <li><b>비상 플랜</b>: 장애 시 폴백(룰기반/평균), 롤백 절차</li>
      </ul>
      <div class="small-muted">※ 운영 모니터링 보드에서는 ‘주말/공휴일’과 ‘평일’을 분리해 추세를 비교하세요.</div>
    </div>
    """
    return ui.HTML(html)
