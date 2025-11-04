# ============================ modules/page_appendix.py (CSS-scoped) ============================
from shiny import ui, render, reactive
from shared import report_df

# ---- Tab: 개요
from viz.appendix_overview import (
    render_data_head,
    render_data_schema,
)

# ---- Tab: EDA (synced with viz/appendix_eda.py)
from viz.appendix_eda import (
    # 데이터 품질/정합성
    render_calendar_alignment_storyline,
    render_calendar_overlay,
    render_midnight_rollover_fix,
    # 기초 통계/결측/이상치
    render_basic_stats,
    render_missing_summary,
    render_outlier_summary,
    # 시계열 스토리라인(월/일/시간/계절)
    render_eda_storyline_panels,
    # 변수 분석
    plot_distribution,
    plot_correlation_heatmap,
    plot_worktype_distribution,
    plot_worktype_hourly_panels,
    # 파생 피처 근거
    render_lag_window_acf,
    render_holiday_peak_checks
)

# ---- Tab: 전처리
from viz.appendix_preproc import (
    render_pipeline_accordion,
    render_feature_summary,
    render_scaling_info,
    render_leakage_check,
)

# ---- Tab: 모델링
from viz.appendix_modeling import (
    render_model_params,
    render_learning_curve,
)

# ---- Tab: 결과/검증
from viz.appendix_results import (
    render_metrics_table,
    render_shap_summary,
)


# ============================ UI ============================
# NOTE: Entire Appendix wrapped in a unique scope to avoid CSS collisions.
# Use `.apx-scope .your-class { ... }` in appendix.css for safe overrides.

def appendix_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="appendix.css"),
        ui.tags.style(
            """
            /* Minimal safe defaults in-scope */
            .apx-scope { --apx-gap: 12px; }
            .apx-scope .billx-titlebox { margin-bottom: var(--apx-gap); }
            .apx-scope .billx-panel { background: #fff; border: 1px solid #e2e8f0; border-radius: .5rem; padding: 12px; }
            .apx-scope .billx-panel-title { margin: 0 0 8px 0; font-weight: 600; }
            .apx-scope .soft { opacity: .4; }
            """
        ),
        ui.div(
            # ===== scope wrapper =====
            ui.div(
                ui.div(
                    ui.h4("데이터 부록 (Appendix)", class_="billx-title"),
                    ui.span("데이터 사전 및 분석 흐름", class_="billx-sub"),
                    class_="billx-titlebox",
                ),
                class_="billx-ribbon billx apx-header",
            ),

            ui.navset_card_pill(
                # ========= 개요 =========
                ui.nav_panel(
                    "개요",
                    ui.layout_columns(
                        ui.div(
                            ui.h5("프로젝트 개요", class_="billx-panel-title"),
                            ui.div(
                                ui.tags.h6("1. 목적", class_="mt-2 mb-1"),
                                ui.tags.p("공장 전력 사용량/요금 분석 및 예측", class_="ms-1 small"),
                                ui.tags.h6("2. 데이터 기간", class_="mt-3 mb-1"),
                                ui.tags.p("2024년 1월 ~ 11월", class_="ms-1 small"),
                                ui.tags.h6("3. 측정 간격", class_="mt-3 mb-1"),
                                ui.tags.p("15분 단위 (일 96개 레코드)", class_="ms-1 small"),
                                ui.tags.h6("4. 예측 타겟", class_="mt-3 mb-1"),
                                ui.tags.p("전기요금(원)", class_="ms-1 fw-bold text-primary"),
                                ui.tags.h6("5. 주요 입력 변수", class_="mt-3 mb-1"),
                                ui.tags.ul(
                                    ui.tags.li("전력사용량(kWh)"),
                                    ui.tags.li("지상/진상 무효전력량(kVarh)"),
                                    ui.tags.li("지상/진상 역률(%)"),
                                    ui.tags.li("탄소배출량(tCO2)"),
                                    ui.tags.li("작업유형"),
                                    class_="ms-2",
                                ),
                                class_="billx-panel-body",
                            ),
                            class_="billx-panel",
                        ),
                        ui.div(
                            ui.h5("데이터 사전 (Data Dictionary)", class_="billx-panel-title"),
                            ui.output_ui("apx_schema_table"),
                            class_="billx-panel",
                        ),
                        col_widths=[5, 7],
                    ),
                    ui.div(
                        ui.h5("데이터 스냅샷 (상위 10행)", class_="billx-panel-title"),
                        ui.output_ui("apx_head_table"),
                        class_="billx-panel",
                    ),
                ),

                # ========= EDA =========
                ui.nav_panel(
                    "EDA",

                    # === 1. 데이터 정합성 ===
                    ui.div(ui.h5("데이터 정합성 검증", class_="billx-panel-title"), class_="billx-panel"),
                    ui.output_ui("apx_calendar_alignment"),

                    ui.div(
                        ui.layout_columns(
                            ui.input_select(
                                "cal_year", "기준 연도 선택",
                                {"2018":"2018","2019":"2019","2021":"2021","2022":"2022","2023":"2023"},
                                selected="2018"
                            ),
                            ui.input_checkbox_group(
                                "cal_mark", "하이라이트 항목",
                                {"weekend":"주말","holiday":"공휴일"},
                                selected=["weekend","holiday"],
                                inline=True,
                            ),
                            col_widths=[4, 8]
                        ),
                        ui.output_ui("apx_calendar_overlay"),
                        class_="mb-3"
                    ),

                    ui.output_ui("apx_midnight_rollover"),

                    ui.hr({"class": "soft"}),

                    # === 2. 기초 통계 & 품질 ===
                    ui.div(ui.h5("기초 통계 & 데이터 품질", class_="billx-panel-title"), class_="billx-panel"),
                    ui.div(ui.output_ui("apx_basic_stats"), class_="billx-panel"),

                    ui.layout_columns(
                        ui.div(ui.h5("결측치 점검", class_="billx-panel-title"), ui.output_ui("apx_missing_summary"), class_="billx-panel"),
                        ui.div(ui.h5("이상치 처리", class_="billx-panel-title"), ui.output_ui("apx_outlier_summary"), class_="billx-panel"),
                        col_widths=[5, 7],
                    ),

                    ui.hr({"class": "soft"}),

                    # === 3. 시계열 스토리라인(중복 제거 버전) ===
                    ui.output_ui("apx_eda_storyline"),

                    ui.hr({"class": "soft"}),

                    # === 4. 변수 분석 ===
                    ui.div(ui.h5("변수 분석", class_="billx-panel-title"), class_="billx-panel"),
                    ui.layout_columns(
                        ui.div(ui.h5("변수 간 상관관계", class_="billx-panel-title"), ui.output_ui("apx_corr_heatmap"), class_="billx-panel"),
                        ui.div(ui.h5("주요 변수 분포", class_="billx-panel-title"), ui.output_ui("apx_dist_plot"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.div(ui.h5("작업유형 × 시간대 패턴 (kWh/원)", class_="billx-panel-title"), ui.output_ui("apx_worktype_hourly"), class_="billx-panel"),
                        ui.div(ui.h5("작업유형별 분포", class_="billx-panel-title"), ui.output_ui("apx_worktype_dist"), class_="billx-panel"),
                        col_widths=[8, 4],
                    ),

                    ui.hr({"class": "soft"}),

                    # === 5. 파생 피처 설계 근거 ===
                    ui.div(
                        ui.h5("추가 설계 근거", class_="billx-panel-title"),
                        ui.div("모델 성능 향상을 위한 파생 피처 설계의 통계적 타당성을 검증합니다.", class_="alert alert-info mb-0"),
                        class_="billx-panel",
                    ),
                    ui.layout_columns(
                        ui.div(ui.h5("시차 상관관계 (ACF)", class_="billx-panel-title"), ui.output_ui("apx_lag_acf"), class_="billx-panel"),
                        ui.div(ui.h5("피크시간대 영향", class_="billx-panel-title"), ui.output_ui("apx_holiday_peak"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),
                ),

                # ========= 전처리 =========
                ui.nav_panel(
                    "전처리",
                    ui.div(ui.h5("전처리 파이프라인", class_="billx-panel-title"), ui.output_ui("apx_pipeline_accordion"), class_="billx-panel"),
                    ui.div(ui.h5("생성된 피처 요약", class_="billx-panel-title"), ui.output_ui("apx_feature_summary"), class_="billx-panel"),
                    ui.layout_columns(
                        ui.div(ui.h5("스케일링/인코딩 전략", class_="billx-panel-title"), ui.output_ui("apx_scaling_info"), class_="billx-panel"),
                        ui.div(ui.h5("데이터 누수 점검", class_="billx-panel-title"), ui.output_ui("apx_leakage_check"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),
                ),

                # ========= 모델링 =========
                ui.nav_panel(
                    "모델링",
                    ui.layout_columns(
                        ui.accordion(
                            ui.accordion_panel(
                                "학습 전략",
                                ui.tags.ul(
                                    ui.tags.li("TimeSeriesSplit(3) → 마지막 폴드 검증"),
                                    ui.tags.li("최신 시점 홀드아웃 유지"),
                                ),
                            ),
                            ui.accordion_panel(
                                "모델 구성",
                                ui.tags.ul(
                                    ui.tags.li("LightGBM (Raw Target)"),
                                    ui.tags.li("XGBoost"),
                                    ui.tags.li("HistGradientBoostingRegressor"),
                                    ui.tags.li("LightGBM (Log Target)"),
                                ),
                            ),
                            ui.accordion_panel(
                                "튜닝 방식",
                                ui.tags.ul(
                                    ui.tags.li("모델별 Random Search"),
                                    ui.tags.li("최적 하이퍼파라미터 탐색 (Holdout MAE 기준)"),
                                ),
                            ),
                            ui.accordion_panel(
                                "평가 기준",
                                ui.tags.ul(
                                    ui.tags.li("Holdout MAE (원 단위)"),
                                ),
                            ),
                            ui.accordion_panel(
                                "앙상블 전략",
                                ui.tags.ul(
                                    ui.tags.li("NNLS 가중치 추정"),
                                    ui.tags.li("NNLS 성능이 단일 모델보다 낮으면 최고 단일 모델 사용"),
                                    ui.tags.li("최종 가중치: LGBM_RAW 0.157, XGB 0.684, HGBR 0.000, LGBM_LOG 0.158 (Holdout MAE 1028.07)"),
                                ),
                            ),
                            ui.accordion_panel(
                                "학습/검증 곡선",
                                ui.div(
                                    ui.input_action_button(
                                        "apx_curve_btn_lgb_raw",
                                        "LGBM (RAW)",
                                        class_="btn btn-outline-primary"
                                    ),
                                    ui.input_action_button(
                                        "apx_curve_btn_xgb",
                                        "XGBoost",
                                        class_="btn btn-outline-primary"
                                    ),
                                    ui.input_action_button(
                                        "apx_curve_btn_hgbr",
                                        "HGBR",
                                        class_="btn btn-outline-primary"
                                    ),
                                    ui.input_action_button(
                                        "apx_curve_btn_lgb_log",
                                        "LGBM (LOG)",
                                        class_="btn btn-outline-primary"
                                    ),
                                    class_="d-flex flex-wrap gap-2 mb-3"
                                ),
                                # ui.output_ui("apx_learning_curve"),
                            ),
                            id="apx_modeling_summary",
                            multiple=True,
                            open=["학습/검증 곡선"],
                        ),
                        ui.div(ui.h5("최종 모델 파라미터", class_="billx-panel-title"), ui.output_ui("apx_model_params"), class_="billx-panel"),
                        col_widths=[7, 5],
                    ),
                    ui.div(ui.h5("학습/검증 곡선", class_="billx-panel-title"), ui.output_ui("apx_learning_curve"), class_="billx-panel"),
                ),

                # ========= 결과/검증 =========
                ui.nav_panel(
                    "결과/검증",
                    ui.layout_columns(
                        ui.div(
                            ui.h5("모델별 잔차 분포", class_="billx-panel-title"),
                            ui.output_ui("apx_shap_summary"),
                            class_="billx-panel"
                        ),
                        ui.div(
                            ui.h5("지상역률 분포 비교", class_="billx-panel-title"),
                            ui.output_ui("apx_metrics_table"),
                            class_="billx-panel"
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                id="apx_tabs",
            ),
            class_="apx-scope",
        ),
    )


# ============================ Server ============================

def appendix_server(input, output, session):
    selected_curve_model = reactive.Value("LGBM (RAW)")

    @reactive.Effect
    @reactive.event(input.apx_curve_btn_lgb_raw)
    def _curve_select_lgb_raw():
        selected_curve_model.set("LGBM (RAW)")

    @reactive.Effect
    @reactive.event(input.apx_curve_btn_xgb)
    def _curve_select_xgb():
        selected_curve_model.set("XGBoost")

    @reactive.Effect
    @reactive.event(input.apx_curve_btn_hgbr)
    def _curve_select_hgbr():
        selected_curve_model.set("HGBR")

    @reactive.Effect
    @reactive.event(input.apx_curve_btn_lgb_log)
    def _curve_select_lgb_log():
        selected_curve_model.set("LGBM (LOG)")

    # ---- 개요
    @output
    @render.ui
    def apx_schema_table():
        return render_data_schema()

    @output
    @render.ui
    def apx_head_table():
        return render_data_head(report_df, n=10)

    # ---- EDA
    @output
    @render.ui
    def apx_calendar_alignment():
        return render_calendar_alignment_storyline(report_df)

    @output
    @render.ui
    def apx_calendar_overlay():
        year = int(input.cal_year() or 2018)
        mark = set(input.cal_mark() or [])
        return render_calendar_overlay(
            report_df,
            year,
            highlight_weekend=("weekend" in mark),
            highlight_holiday=("holiday" in mark),
        )

    @output
    @render.ui
    def apx_midnight_rollover():
        return render_midnight_rollover_fix(report_df)

    @output
    @render.ui
    def apx_basic_stats():
        return render_basic_stats(report_df)

    @output
    @render.ui
    def apx_missing_summary():
        return render_missing_summary(report_df)

    @output
    @render.ui
    def apx_outlier_summary():
        return render_outlier_summary(report_df)

    @output
    @render.ui
    def apx_eda_storyline():
        return render_eda_storyline_panels(report_df)

    @output
    @render.ui
    def apx_corr_heatmap():
        return plot_correlation_heatmap(report_df)

    @output
    @render.ui
    def apx_dist_plot():
        return plot_distribution(report_df)

    @output
    @render.ui
    def apx_worktype_hourly():
        return plot_worktype_hourly_panels(report_df)

    @output
    @render.ui
    def apx_worktype_dist():
        return plot_worktype_distribution(report_df)

    # ---- 파생 피처 근거
    @output
    @render.ui
    def apx_lag_acf():
        return render_lag_window_acf(report_df)

    @output
    @render.ui
    def apx_holiday_peak():
        return render_holiday_peak_checks(report_df)

        # ---- 전처리
    @output
    @render.ui
    def apx_pipeline_accordion():
        return render_pipeline_accordion()

    @output
    @render.ui
    def apx_feature_summary():
        return render_feature_summary()

    @output
    @render.ui
    def apx_scaling_info():
        return render_scaling_info()

    @output
    @render.ui
    def apx_leakage_check():
        return render_leakage_check()

    # ---- 모델링
    @output
    @render.ui
    def apx_model_params():
        return render_model_params()

    @output
    @render.ui
    def apx_learning_curve():
        return render_learning_curve(selected_curve_model())

    # ---- 결과/검증
    @output
    @render.ui
    def apx_metrics_table():
        return render_metrics_table()

    @render.ui
    def apx_shap_summary():
        return render_shap_summary()

