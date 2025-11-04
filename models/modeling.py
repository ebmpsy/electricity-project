from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import nnls as SCIPY_NNLS  # 비음이 아닌 최소제곱(Non-negative LS) → 앙상블 가중치 추정

import lightgbm as lgb
import xgboost as xgb
import json

# =======================================================================================================================================
# 경로
# =======================================================================================================================================
BASE_DIR = Path(r"C:\Users\USER\Desktop\ELECTRICITY-PROJECT")
TRAIN_PATH = BASE_DIR / 'data' / 'train.csv'
TEST_PATH = BASE_DIR / 'data' / 'test.csv'
SUB_PATH = BASE_DIR / 'data' / 'output' / 'submission.csv'
CURVE_DIR = BASE_DIR / 'data' / 'output'
CURVE_DIR.mkdir(parents=True, exist_ok=True)

# =======================================================================================================================================
# 상수
# =======================================================================================================================================
TARGET = "전기요금(원)"
DT_COL = "측정일시"
CAT_COL = "작업유형"
RANDOM_STATE = 42

# 하이퍼파라미터 탐색 시 몇 번 샘플링할지
LGBM_SEARCH_ITER = 28
LGBM_LOG_SEARCH_ITER = 20
XGB_SEARCH_ITER = 32
HGBR_SEARCH_ITER = 24

# =======================================================================================================================================
# 유틸 함수
# =======================================================================================================================================
def fix_midnight_rollover(df, dt_col):
    """
    00:00 시각이 다음 날로 넘어가야 하는데 안 넘어간 데이터 보정해주는 함수.
    """
    out = df.copy()
    dt = pd.to_datetime(out[dt_col])
    mask = (dt.dt.hour == 0) & (dt.dt.minute == 0)
    out.loc[mask, dt_col] = dt.loc[mask] + pd.Timedelta(days=1)
    return out

def get_korean_holidays_2018():
    """
    2018년에 해당하는 한국 공휴일 세트를 만들어서 반환.
    """
    days = set()
    add = days.add

    def add_range(start, end):
        for dt in pd.date_range(start, end, freq="D"):
            add(dt.date())

    add(pd.to_datetime("2018-01-01").date())  # 신정
    add_range("2018-02-15", "2018-02-17")     # 설날 연휴
    add(pd.to_datetime("2018-03-01").date())  # 삼일절
    add(pd.to_datetime("2018-05-07").date())  # 어린이날 대체공휴일
    add(pd.to_datetime("2018-05-22").date())  # 부처님오신날
    add(pd.to_datetime("2018-06-06").date())  # 현충일
    add(pd.to_datetime("2018-06-13").date())  # 지방선거
    add(pd.to_datetime("2018-08-15").date())  # 광복절
    add_range("2018-09-22", "2018-09-26")     # 추석 연휴 및 대체공휴일
    add(pd.to_datetime("2018-10-03").date())  # 개천절
    add(pd.to_datetime("2018-10-09").date())  # 한글날
    add(pd.to_datetime("2018-12-25").date())  # 성탄절
    add(pd.to_datetime("2018-12-31").date())  # 연말
    return days

HOLIDAYS_2018 = get_korean_holidays_2018()

def base_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본적인 시간 관련 파생변수들을 만드는 함수
    """
    out = df.copy()
    dt = pd.to_datetime(out[DT_COL])

    # 기본 날짜/시간 단위
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["minute"] = dt.dt.minute
    out["weekday"] = dt.dt.weekday  # 월=0, ..., 일=6
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["month"] = dt.dt.month

    # 공휴일 관련 플래그
    ddate = dt.dt.date
    out["is_holiday"] = [1 if d in HOLIDAYS_2018 else 0 for d in ddate]
    out["is_holiday_eve"] = [1 if (pd.Timestamp(d) - pd.Timedelta(days=1)).date() in HOLIDAYS_2018 else 0 for d in ddate]
    out["is_holiday_after"] = [1 if (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in HOLIDAYS_2018 else 0 for d in ddate]

    # 주기항 (시간 단위 24h 기준, 요일 단위 7d 기준)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    # 2고조파
    out["hour_sin2"] = np.sin(4 * np.pi * out["hour"] / 24)
    out["hour_cos2"] = np.cos(4 * np.pi * out["hour"] / 24)
    # 요일 사이클
    out["dow_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)

    # 전력 사용량이 급증할 법한 시간대 플래그
    out["is_peak_after"] = ((out["hour"] >= 13) & (out["hour"] <= 17)).astype(int)
    out["is_peak_even"] = ((out["hour"] >= 18) & (out["hour"] <= 22)).astype(int)
    out["is_night"] = ((out["hour"] >= 23) | (out["hour"] <= 5)).astype(int)

    # 주 + 시간 → 0~167
    out["hour_of_week"] = out["weekday"] * 24 + out["hour"]

    # CAT_COL 카테고리화
    out[CAT_COL] = out[CAT_COL].astype('category')
    return out

def add_factory_holiday_flag(train_f: pd.DataFrame,
                             test_f: pd.DataFrame,
                             threshold: float = 15.0):
    """
    oof_kwh 기준으로 날짜별 평균을 내고,
    그 평균이 threshold 이하인 날짜를 '공장 휴일'로 보는 플래그를 추가한다.
    train/test가 둘 다 2018년 타임라인을 쓰니까, 두 개를 합쳐서 같은 기준으로 잡는다.
    """
    # train / test 붙여서 같은 기준으로 판단
    tmp = pd.concat(
        [train_f.assign(_split='train'), test_f.assign(_split='test')],
        ignore_index=True
    ).copy()

    # 날짜만 뽑기
    tmp['_date'] = pd.to_datetime(tmp[DT_COL]).dt.date

    # 일별 oof_kwh 평균
    daily_mean = tmp.groupby('_date')['oof_kwh'].mean()

    # threshold 이하인 날짜들
    holiday_dates = set(daily_mean[daily_mean <= threshold].index)

    # 플래그 생성
    tmp['is_factory_holiday'] = tmp['_date'].isin(holiday_dates).astype(int)

    # 다시 train/test로 나누기
    train_flag = tmp[tmp['_split'] == 'train']['is_factory_holiday'].to_numpy()
    test_flag = tmp[tmp['_split'] == 'test']['is_factory_holiday'].to_numpy()

    train_f['is_factory_holiday'] = train_flag
    test_f['is_factory_holiday'] = test_flag

    return train_f, test_f

# =======================================================================================================================================
# Step1: OOF 3변수
# =======================================================================================================================================
def oof_3vars(train_f, test_f):
    """
    전력 관련 3개 컬럼을 OOF(TimeSeriesSplit)로 예측해 train/test 둘 다에
    oof_kwh, oof_reactive, oof_pf 컬럼을 추가하는 단계.
    """

    # 시간 기반 피처 중에서 기본적으로 쓰는 subset
    basic_cols = [
        'hour', 'day', 'minute', 'weekday', 'is_weekend',
        'hour_sin', 'hour_cos', 'hour_sin2', 'hour_cos2',
        'dow_sin', 'dow_cos', 'is_holiday', 'is_holiday_eve', 'is_holiday_after',
        'is_peak_after', 'is_peak_even', 'is_night', 'hour_of_week'
    ]

    # 작업유형을 숫자로 인코딩해서 같이 넣어줌
    le = LabelEncoder()
    train_f['cat_encoded'] = le.fit_transform(train_f[CAT_COL])
    test_f['cat_encoded'] = le.transform(test_f[CAT_COL])
    basic_cols += ['cat_encoded']

    Xtr = train_f[basic_cols]
    Xte = test_f[basic_cols]

    # 시계열 분할
    tscv = TimeSeriesSplit(n_splits=5)

    def run_one(y):
        """
        하나의 타겟(예: 전력사용량)에 대해 5개 시계열 폴드로
        - train_f 부분에 대해서는 OOF 예측값
        - test_f 전체에 대해서는 5모델 평균 예측값
        을 만들어주는 내부 함수
        """
        oof = np.zeros(len(Xtr))
        pred = np.zeros(len(Xte))
        for tr_idx, va_idx in tscv.split(Xtr):
            # 여기서는 간단한 LGBM 하나만 쓴다 (하이퍼 고정)
            model = lgb.LGBMRegressor(
                objective="regression_l1",
                n_estimators=700,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                max_depth=-1,
                min_child_samples=25,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(Xtr.iloc[tr_idx], y.iloc[tr_idx])
            # OOF 채우기
            oof[va_idx] = model.predict(Xtr.iloc[va_idx])
            # test 예측은 폴드마다 더해서 평균
            pred += model.predict(Xte) / 5.0
        return oof, pred

    # 원본 데이터 안에 있으면 잡아서 OOF화 할 3개 컬럼
    targets = {
        '전력사용량(kWh)': 'oof_kwh',
        '지상무효전력량(kVarh)': 'oof_reactive',
        '지상역률(%)': 'oof_pf'
    }

    for raw, newc in targets.items():
        if raw in train_f.columns:
            y = pd.to_numeric(train_f[raw], errors='coerce').fillna(0)
            o, p = run_one(y)
            # 음수가 나올 수 있는 컬럼도 있어서 최소 0으로 clip
            train_f[newc] = o.clip(min=0)
            test_f[newc] = p.clip(min=0)
        else:
            # 만약 원본 열이 없으면 NaN으로 채워서 후단에서 알아서 처리하게 함
            train_f[newc] = np.nan
            test_f[newc] = np.nan

    return train_f, test_f

# =======================================================================================================================================
# Step2: v1+ 파생
# =======================================================================================================================================
def deriv_from_oof(df: pd.DataFrame) -> pd.DataFrame:
    """
    위에서 만든 oof_kwh / oof_reactive / oof_pf를 가지고
    시계열 파생 컬럼을 여러 개 만드는 함수.
    """
    out = df.copy().sort_values(DT_COL)

    # 원 코드 기준으로 step이 15분 단위 같은 디테일이 있는 듯: 4 = 1시간(15분*4)
    step_1h, step_6h, step_24h = 4, 24, 96
    step_48h, step_7d, step_14d = 192, 7 * 24 * 4, 14 * 24 * 4

    cols = ['oof_kwh', 'oof_reactive', 'oof_pf']
    for col in cols:
        s = pd.to_numeric(out[col], errors='coerce')

        # 여러 종류의 시차 컬럼
        out[f'{col}_lag1'] = s.shift(1)
        out[f'{col}_lag1h'] = s.shift(step_1h)
        out[f'{col}_lag6h'] = s.shift(step_6h)
        out[f'{col}_lag24h'] = s.shift(step_24h)
        out[f'{col}_lag48h'] = s.shift(step_48h)    # ✅ 추가된 48시간 lag
        out[f'{col}_lag7d'] = s.shift(step_7d)

        # 과거 평균/ema
        out[f'{col}_roll6h'] = s.shift(1).rolling(step_6h, min_periods=3).mean()
        out[f'{col}_roll24h'] = s.shift(1).rolling(step_24h, min_periods=6).mean()
        out[f'{col}_ema24h'] = s.shift(1).ewm(span=step_24h, adjust=False, min_periods=6).mean()

        # 같은 시각끼리의 차이 (하루 전 / 1주 전 / 2주 전)
        out[f'{col}_samehour_d1'] = s - s.shift(step_24h)
        out[f'{col}_samehour_w1'] = s - s.shift(step_7d)
        out[f'{col}_samehour_w2'] = s - s.shift(step_14d)  # ✅ 2주 전과 비교
        out[f'{col}_diff1h'] = s.diff(step_1h)

    # 추가적인 비율/역률 유도 피처
    a = pd.to_numeric(out['oof_kwh'], errors='coerce')
    r = pd.to_numeric(out['oof_reactive'], errors='coerce')
    out['oof_ratio'] = (r / (a + 1e-6)).replace([np.inf, -np.inf], np.nan)
    out['oof_pf_proxy'] = a / (a + r + 1e-6)  # 실역률이 없거나 이상할 때 쓸 수 있는 간이 지표
    out['oof_ratio_ema24h'] = out['oof_ratio'].shift(1).ewm(
        span=step_24h, adjust=False, min_periods=6
    ).mean()

    return out

# =======================================================================================================================================
# Step3: 시간대 프로필 잔차 & 변화율
# =======================================================================================================================================
def add_seasonal_profile_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    hour_of_week(0~167)별로 '이 시간대는 보통 이만큼 쓴다'는 프로필을 만들고,
    실제 oof_kwh와의 차이를 잔차로 두는 함수.
    → 주간 시즌성 잡으려는 의도.

    또, 1주/2주 변화율도 같이 만든다.
    """
    out = df.copy().sort_values(DT_COL)

    how = pd.to_numeric(out["hour_of_week"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(out["oof_kwh"], errors="coerce")

    # 시간대별 프로필을 담을 시리즈
    prof = pd.Series(np.nan, index=out.index)

    # 0~167 각각에 대해서, 그 시간대의 과거값 평균(누적)으로 프로필을 만듦
    # cumsum / count 를 한 칸씩 밀어주면 해당 시점 이전까지의 평균이 됨 → OOF 비슷한 효과
    for h in range(168):
        idx = (how == h)
        seq = s.where(idx)
        csum = seq.cumsum()
        ccnt = idx.cumsum()
        # 바로 직전까지의 평균만 쓰고 싶으므로 shift(1)
        prof[idx] = (csum.shift(1) / ccnt.shift(1)).where(ccnt.shift(1) > 0, np.nan)

    out["how_profile_kwh"] = prof
    out["how_resid_kwh"] = s - prof  # 실제값 - 프로필 = 잔차

    # 변화율: 일주일 전/2주 전 대비 % 변화
    step_7d = 7 * 24 * 4
    step_14d = 14 * 24 * 4
    eps = 1e-6

    out["kwh_rate_w1"] = ((s - s.shift(step_7d)) / (np.abs(s.shift(step_7d)) + eps)).clip(-3, 3)
    out["kwh_rate_w2"] = ((s - s.shift(step_14d)) / (np.abs(s.shift(step_14d)) + eps)).clip(-3, 3)

    return out


# =======================================================================================================================================
# 메인
# =======================================================================================================================================

# 데이터 로드
train = pd.read_csv(TRAIN_PATH, parse_dates=[DT_COL])
test = pd.read_csv(TEST_PATH, parse_dates=[DT_COL])

train_dt = pd.to_datetime(train[DT_COL])
test_dt = pd.to_datetime(test[DT_COL])

# 2018년도로 맵핑
train[DT_COL] = train_dt.apply(
    lambda ts: ts - pd.DateOffset(years=ts.year - 2018) if ts.year != 2018 else ts
)
test[DT_COL] = test_dt.apply(
    lambda ts: ts - pd.DateOffset(years=ts.year - 2018) if ts.year != 2018 else ts
)

# 00:00 보정
train = fix_midnight_rollover(train, DT_COL)
test = fix_midnight_rollover(test, DT_COL)

# 알려진 이상 시점 제거 (문제에서 제공된 것 같음)
filter_time = pd.Timestamp("2018-11-07 00:00:00")
train = train[train[DT_COL] != filter_time]

# 시계열 정렬
train = train.sort_values(DT_COL).reset_index(drop=True)
test = test.sort_values(DT_COL).reset_index(drop=True)

# =======================================================================================================================================
# 파생
# =======================================================================================================================================
# 기본 시간 피처
train_f = base_time_feats(train)
test_f = base_time_feats(test)

# OOF 3변수
train_f, test_f = oof_3vars(train_f, test_f)

# OOF 기반 공장 휴일 플래그
train_f, test_f = add_factory_holiday_flag(train_f, test_f)

# OOF 파생 피처
train_f = deriv_from_oof(train_f)
test_f = deriv_from_oof(test_f)

# 시간대 프로필 잔차 & 변화율
train_f = add_seasonal_profile_residuals(train_f)
test_f = add_seasonal_profile_residuals(test_f)

# hour_of_week를 범주형으로도 하나 더 들고감 (LightGBM categorical)
train_f["how_cat"] = train_f["hour_of_week"].astype("string")
test_f["how_cat"] = test_f["hour_of_week"].astype("string")

# =======================================================================================================================================
# 결측 처리
# =======================================================================================================================================
# 원본 전력 관련 컬럼은 후단에서 사용하지 않으려고 제외하려는 것
raw_power_cols = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)', '지상역률(%)', '탄소배출량(tCO2)']

# 학습용 수치 피처 후보: 날짜/타겟/범주 컬럼/원본 전력컬럼 제외 + 숫자형
num_features_train = [
    c for c in train_f.columns
    if (c not in [DT_COL, TARGET, CAT_COL, 'how_cat'])
    and pd.api.types.is_numeric_dtype(train_f[c])
    and (c not in raw_power_cols)
]

# train과 test가 공통으로 가지는 것만 사용
num_features = [c for c in num_features_train if c in test_f.columns]

# 범주형 피처 정의
cat_features = [CAT_COL, 'how_cat']
all_features = num_features + cat_features

# 수치형 결측치는 train의 중앙값으로 채우고, 같은 값으로 test도 채움
median_map = {c: pd.to_numeric(train_f[c], errors='coerce').median(skipna=True) for c in num_features}
for c in num_features:
    train_f[c] = pd.to_numeric(train_f[c], errors='coerce').fillna(median_map[c])
    test_f[c] = pd.to_numeric(test_f[c], errors='coerce').fillna(median_map[c])

# 범주형은 문자열로 맞추고 train/test 합집합으로 카테고리 통일
train_f['how_cat'] = train_f['how_cat'].astype('string')
test_f['how_cat'] = test_f['how_cat'].astype('string')

for col in cat_features:
    train_f[col] = train_f[col].astype('string').fillna('UNK')
    test_f[col] = test_f[col].astype('string').fillna('UNK')
    cats = pd.Index(pd.unique(pd.concat([train_f[col], test_f[col]], ignore_index=True))).astype(str)
    train_f[col] = pd.Categorical(train_f[col], categories=cats)
    test_f[col] = pd.Categorical(test_f[col], categories=cats)

# =======================================================================================================================================
# 최종 데이터셋 준비
# =======================================================================================================================================
X_train = train_f[all_features].copy()
X_test = test_f[all_features].copy()

# 타겟을 수치로 캐스팅하고 0 미만은 0으로
y_raw = pd.to_numeric(train[TARGET], errors='coerce').fillna(0).clip(lower=0).values
# 로그 타겟도 같이 만들어둠 (XGB, HGBR, LGB log-target에서 사용)
y_log = np.log1p(y_raw)
# 역변환 함수 (안전하게 상한 20 걸어둠)
inv = lambda p: np.expm1(np.clip(p, None, 30))

# ===================== CV =====================
# 시계열 3분할 → 마지막 폴드를 홀드아웃으로 쓰는 구조
tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X_train))
tr_idx, va_idx = splits[-1]
print(f"\n[5] 3-Fold TimeSeriesSplit (Holdout: {len(va_idx)})")

# 가중치는 일단 1로 고정
w = np.ones(len(X_train))

# 모델별 예측을 저장할 dict
pred, hold = {}, {}
best_param_records = []

# =======================================================================================================================================
# 모델링
# =======================================================================================================================================
# -------------------------------------------------
# 1) LightGBM (원시 타겟 기준)
# -------------------------------------------------
print("\n[6-1] LightGBM (RAW Target)")

# 공통 파라미터
lgb_base_params = dict(
    objective='regression_l1',  # MAE 계열
    metric='mae',
    random_state=RANDOM_STATE
)

# 탐색할 파라미터 공간
lgb_param_space = {
    "num_leaves": [40, 48, 56, 64, 72, 96],
    "learning_rate": [0.025, 0.03, 0.035, 0.04, 0.045],
    "feature_fraction": [0.75, 0.8, 0.85, 0.9],
    "bagging_fraction": [0.75, 0.8, 0.85, 0.9],
    "min_child_samples": [30, 45, 60, 75, 90],
    "reg_alpha": [0.0, 0.05, 0.1, 0.15, 0.2],
    "reg_lambda": [0.1, 0.2, 0.35, 0.5, 0.8],
    "num_boost_round": [1200, 1500, 1800],
    "early_stopping_rounds": [90, 120, 150]
}
# 랜덤하게 n_iter개만 뽑아서 빠르게 탐색
lgb_samples = list(ParameterSampler(lgb_param_space, n_iter=LGBM_SEARCH_ITER, random_state=RANDOM_STATE))
print(f"  탐색 조합: {len(lgb_samples)}개")

best_lgb_raw = {"mae": np.inf, "params": None, "model": None}

for idx, params in enumerate(lgb_samples, 1):
    params = params.copy()
    num_round = params.pop("num_boost_round")
    early_stop = params.pop("early_stopping_rounds")

    trial_params = {**lgb_base_params, **params}

    # LightGBM은 별도의 Dataset 객체로 범주형을 지정해줘야 함
    train_data = lgb.Dataset(
        X_train.iloc[tr_idx],
        label=y_raw[tr_idx],
        weight=w[tr_idx],
        categorical_feature=cat_features
    )
    valid_data = lgb.Dataset(
        X_train.iloc[va_idx],
        label=y_raw[va_idx],
        weight=w[va_idx],
        categorical_feature=cat_features,
        reference=train_data
    )

    model = lgb.train(
        trial_params,
        train_data,
        num_boost_round=num_round,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(early_stop, first_metric_only=True),
            lgb.log_evaluation(0)  # 출력 줄임
        ]
    )

    # 홀드아웃 예측 후 MAE 계산
    val_pred = model.predict(X_train.iloc[va_idx], num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_raw[va_idx], val_pred)

    # 더 좋으면 갱신
    if mae < best_lgb_raw["mae"]:
        best_lgb_raw.update({
            "mae": mae,
            "params": {**trial_params, "num_boost_round": num_round, "early_stopping_rounds": early_stop},
            "model": model
        })

print(f"  Best Holdout MAE: {best_lgb_raw['mae']:.2f}")
best_params_raw = best_lgb_raw["params"].copy()
display_params_raw = {k: best_params_raw[k] for k in best_params_raw if k not in ('objective', 'metric', 'random_state')}
print(f"  Best Params: {display_params_raw}")
best_param_records.append({
    "model": "LGBM_RAW",
    "mae": best_lgb_raw["mae"],
    "params": json.dumps(best_params_raw, ensure_ascii=False)
})

# 베스트 파라미터로 다시 한 번 학습해서 test에도 예측
num_round = best_params_raw.pop("num_boost_round")
early_stop = best_params_raw.pop("early_stopping_rounds")

lgb_tr = lgb.Dataset(
    X_train.iloc[tr_idx], label=y_raw[tr_idx], weight=w[tr_idx], categorical_feature=cat_features
)
lgb_va = lgb.Dataset(
    X_train.iloc[va_idx], label=y_raw[va_idx], weight=w[va_idx], categorical_feature=cat_features, reference=lgb_tr
)

lgb_eval_result_raw = {}
lgb_model = lgb.train(
    best_params_raw,
    lgb_tr,
    num_boost_round=num_round,
    valid_sets=[lgb_va, lgb_tr],
    valid_names=['valid', 'train'],
    callbacks=[
        lgb.early_stopping(early_stop, first_metric_only=True),
        lgb.record_evaluation(lgb_eval_result_raw),
        lgb.log_evaluation(0)
    ]
)

metric_name_raw = next(iter(lgb_eval_result_raw['valid']))
lgb_raw_curve = pd.DataFrame({
    'iteration': np.arange(1, len(lgb_eval_result_raw['valid'][metric_name_raw]) + 1),
    'valid_' + metric_name_raw: lgb_eval_result_raw['valid'][metric_name_raw],
    'train_' + metric_name_raw: lgb_eval_result_raw['train'][metric_name_raw]
})
raw_curve_path = CURVE_DIR / 'lgb_raw_learning_curve.csv'
lgb_raw_curve.to_csv(raw_curve_path, index=False)
print(f"  Learning curve saved: {raw_curve_path}")

# test 예측과 홀드아웃 예측 저장
pred['LGBM_RAW'] = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
hold['LGBM_RAW'] = lgb_model.predict(X_train.iloc[va_idx], num_iteration=lgb_model.best_iteration)
print(f"  Holdout MAE (Raw): {mean_absolute_error(y_raw[va_idx], hold['LGBM_RAW']):.2f}")

# -------------------------------------------------
# 2) XGBoost (로그 타겟으로 학습 후 역변환)
# -------------------------------------------------
print("\n[6-2] XGBoost")

# XGBoost는 카테고리를 숫자 코드로 직접 넘겨줘야 함
Xtr = X_train.copy()
Xte = X_test.copy()
for c in cat_features:
    Xtr[c] = Xtr[c].cat.codes
    Xte[c] = Xte[c].cat.codes

# 랜덤 탐색 공간
xgb_param_space = {
    "max_depth": [6, 7, 8, 9, 10],
    "learning_rate": [0.025, 0.03, 0.035, 0.04, 0.045],
    "subsample": [0.7, 0.75, 0.8, 0.85, 0.9],
    "colsample_bytree": [0.7, 0.75, 0.8, 0.85, 0.9],
    "reg_lambda": [0.8, 1.0, 1.2, 1.5, 2.0],
    "reg_alpha": [0.0, 0.1, 0.2, 0.3, 0.4],
    "min_child_weight": [1, 2, 3, 4],
    "gamma": [0.0, 0.05, 0.1, 0.2],
    "n_estimators": [1100, 1400, 1700, 2000],
    "early_stopping_rounds": [80, 100, 120, 150]
}
xgb_samples = list(ParameterSampler(xgb_param_space, n_iter=XGB_SEARCH_ITER, random_state=RANDOM_STATE + 2))
print(f"  탐색 조합: {len(xgb_samples)}개")

best_xgb = {"mae": np.inf, "params": None}

for params in xgb_samples:
    trial = params.copy()
    n_estimators = trial.pop("n_estimators")
    early_stop = trial.pop("early_stopping_rounds")

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stop,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        **trial
    )

    model.fit(
        Xtr.iloc[tr_idx],
        y_log[tr_idx],  # 로그 타겟
        sample_weight=w[tr_idx],
        eval_set=[(Xtr.iloc[va_idx], y_log[va_idx])],
        verbose=False
    )

    # 예측은 역변환해서 MAE 계산
    val_pred = inv(model.predict(Xtr.iloc[va_idx]))
    mae = mean_absolute_error(y_raw[va_idx], val_pred)

    if mae < best_xgb["mae"]:
        best_xgb.update({
            "mae": mae,
            "params": {**trial, "n_estimators": n_estimators, "early_stopping_rounds": early_stop}
        })

print(f"  Best Holdout MAE: {best_xgb['mae']:.2f}")
best_params_xgb = best_xgb["params"].copy()
display_params_xgb = {k: best_params_xgb[k] for k in best_params_xgb}
print(f"  Best Params: {display_params_xgb}")
best_param_records.append({
    "model": "XGB",
    "mae": best_xgb["mae"],
    "params": json.dumps(best_params_xgb, ensure_ascii=False)
})

# 베스트로 다시 학습
n_estimators = best_params_xgb.pop("n_estimators")
early_stop = best_params_xgb.pop("early_stopping_rounds")

xgb_model = xgb.XGBRegressor(
    n_estimators=n_estimators,
    early_stopping_rounds=early_stop,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    eval_metric='mae',
    **best_params_xgb
)
xgb_model.fit(
    Xtr.iloc[tr_idx],
    y_log[tr_idx],
    sample_weight=w[tr_idx],
    eval_set=[
        (Xtr.iloc[tr_idx], y_log[tr_idx]),
        (Xtr.iloc[va_idx], y_log[va_idx])
    ],
    verbose=False
)

xgb_eval_result = xgb_model.evals_result()
train_metric_key = next(iter(xgb_eval_result['validation_0']))
total_rounds = len(xgb_eval_result['validation_0'][train_metric_key])

xgb_curve_records = []
tr_features = Xtr.iloc[tr_idx]
va_features = Xtr.iloc[va_idx]
tr_target = y_raw[tr_idx]
va_target = y_raw[va_idx]
for iter_idx in range(1, total_rounds + 1):
    pred_tr_log = xgb_model.predict(tr_features, iteration_range=(0, iter_idx))
    pred_va_log = xgb_model.predict(va_features, iteration_range=(0, iter_idx))
    pred_tr = inv(pred_tr_log)
    pred_va = inv(pred_va_log)
    mae_tr = mean_absolute_error(tr_target, pred_tr)
    mae_va = mean_absolute_error(va_target, pred_va)
    xgb_curve_records.append((iter_idx, mae_tr, mae_va))

xgb_curve = pd.DataFrame(xgb_curve_records, columns=['iteration', 'train_mae', 'valid_mae'])
xgb_curve_path = CURVE_DIR / 'xgb_learning_curve.csv'
xgb_curve.to_csv(xgb_curve_path, index=False)
print(f"  Learning curve saved: {xgb_curve_path}")

pred['XGB'] = inv(xgb_model.predict(Xte))
hold['XGB'] = inv(xgb_model.predict(Xtr.iloc[va_idx]))
print(f"  Holdout MAE: {mean_absolute_error(y_raw[va_idx], hold['XGB']):.2f}")

# -------------------------------------------------
# 3) HistGradientBoostingRegressor (로그 타겟)
# -------------------------------------------------
print("\n[6-3] HGBR")

hgbr_param_space = {
    "learning_rate": [0.025, 0.03, 0.035, 0.04, 0.045],
    "max_iter": [900, 1200, 1500, 1800],
    "max_depth": [None, 6, 8, 10],
    "max_leaf_nodes": [31, 63, 127, 255],
    "min_samples_leaf": [10, 15, 20, 30, 40],
    "l2_regularization": [0.0, 0.0005, 0.001, 0.005, 0.01],
    "max_bins": [128, 160, 192, 224, 255]
}
hgbr_samples = list(ParameterSampler(hgbr_param_space, n_iter=HGBR_SEARCH_ITER, random_state=RANDOM_STATE + 3))
print(f"  탐색 조합: {len(hgbr_samples)}개")

best_hgbr = {"mae": np.inf, "params": None}

for params in hgbr_samples:
    trial = params.copy()
    max_iter = trial.pop("max_iter")

    model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=False,
        max_iter=max_iter,
        **trial
    )
    model.fit(Xtr.iloc[tr_idx], y_log[tr_idx], sample_weight=w[tr_idx])

    val_pred = inv(model.predict(Xtr.iloc[va_idx]))
    mae = mean_absolute_error(y_raw[va_idx], val_pred)

    if mae < best_hgbr["mae"]:
        best_hgbr.update({
            "mae": mae,
            "params": {**trial, "max_iter": max_iter}
        })

print(f"  Best Holdout MAE: {best_hgbr['mae']:.2f}")
best_params_hgbr = best_hgbr["params"].copy()
display_params_hgbr = {k: best_params_hgbr[k] for k in best_params_hgbr}
print(f"  Best Params: {display_params_hgbr}")
best_param_records.append({
    "model": "HGBR",
    "mae": best_hgbr["mae"],
    "params": json.dumps(best_params_hgbr, ensure_ascii=False)
})

# 베스트로 다시 학습
max_iter = best_params_hgbr.pop("max_iter")
hgbr = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE,
    early_stopping=False,
    max_iter=max_iter,
    **best_params_hgbr
)
hgbr.fit(Xtr.iloc[tr_idx], y_log[tr_idx], sample_weight=w[tr_idx])

# staged_predict로 학습/검증 MAE 기록
train_stage_preds = hgbr.staged_predict(Xtr.iloc[tr_idx])
valid_stage_preds = hgbr.staged_predict(Xtr.iloc[va_idx])
hgbr_curve_records = []
for iter_idx, (pred_tr, pred_va) in enumerate(zip(train_stage_preds, valid_stage_preds), start=1):
    mae_tr = mean_absolute_error(y_raw[tr_idx], inv(pred_tr))
    mae_va = mean_absolute_error(y_raw[va_idx], inv(pred_va))
    hgbr_curve_records.append((iter_idx, mae_tr, mae_va))

if hgbr_curve_records:
    hgbr_curve = pd.DataFrame(hgbr_curve_records, columns=['iteration', 'train_mae', 'valid_mae'])
    hgbr_curve_path = CURVE_DIR / 'hgbr_learning_curve.csv'
    hgbr_curve.to_csv(hgbr_curve_path, index=False)
    print(f"  Learning curve saved: {hgbr_curve_path}")

pred['HGBR'] = inv(hgbr.predict(Xte))
hold['HGBR'] = inv(hgbr.predict(Xtr.iloc[va_idx]))
print(f"  Holdout MAE: {mean_absolute_error(y_raw[va_idx], hold['HGBR']):.2f}")

# -------------------------------------------------
# 4) LightGBM (로그 타겟으로 별도 한 번 더)
# -------------------------------------------------
print("\n[6-4] LightGBM LOG-TARGET (추가)")
lgb_log_base = dict(
    objective='huber',  # 로그 타겟이라도 huber로 MAE 비슷한 효과
    metric='mae',
    random_state=RANDOM_STATE
)

lgb_log_param_space = {
    "num_leaves": [40, 48, 56, 64, 72, 96],
    "learning_rate": [0.025, 0.03, 0.035, 0.04, 0.045],
    "feature_fraction": [0.75, 0.8, 0.85, 0.9],
    "bagging_fraction": [0.75, 0.8, 0.85, 0.9],
    "min_child_samples": [30, 45, 60, 75, 90],
    "reg_alpha": [0.0, 0.05, 0.1, 0.15, 0.2],
    "reg_lambda": [0.1, 0.2, 0.35, 0.5, 0.8],
    "num_boost_round": [1200, 1500, 1800],
    "early_stopping_rounds": [90, 120, 150]
}
lgb_log_samples = list(
    ParameterSampler(lgb_log_param_space, n_iter=LGBM_LOG_SEARCH_ITER, random_state=RANDOM_STATE + 4)
)
print(f"  탐색 조합: {len(lgb_log_samples)}개")

best_lgb_log = {"mae": np.inf, "params": None}

for params in lgb_log_samples:
    params = params.copy()
    num_round = params.pop("num_boost_round")
    early_stop = params.pop("early_stopping_rounds")

    trial_params = {**lgb_log_base, **params}

    train_data = lgb.Dataset(X_train.iloc[tr_idx], label=y_log[tr_idx], categorical_feature=cat_features)
    valid_data = lgb.Dataset(
        X_train.iloc[va_idx], label=y_log[va_idx], categorical_feature=cat_features, reference=train_data
    )

    model = lgb.train(
        trial_params,
        train_data,
        num_boost_round=num_round,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(early_stop, first_metric_only=True), lgb.log_evaluation(0)]
    )

    val_pred = inv(model.predict(X_train.iloc[va_idx], num_iteration=model.best_iteration))
    mae = mean_absolute_error(y_raw[va_idx], val_pred)

    if mae < best_lgb_log["mae"]:
        best_lgb_log.update({
            "mae": mae,
            "params": {**trial_params, "num_boost_round": num_round, "early_stopping_rounds": early_stop}
        })

print(f"  Best Holdout MAE: {best_lgb_log['mae']:.2f}")
best_params_log = best_lgb_log["params"].copy()
display_params_log = {k: best_params_log[k] for k in best_params_log if k not in ('objective', 'metric', 'random_state')}
print(f"  Best Params: {display_params_log}")
best_param_records.append({
    "model": "LGBM_LOG",
    "mae": best_lgb_log["mae"],
    "params": json.dumps(best_params_log, ensure_ascii=False)
})

# 베스트로 다시 학습
num_round = best_params_log.pop("num_boost_round")
early_stop = best_params_log.pop("early_stopping_rounds")

lgb_tr_log = lgb.Dataset(X_train.iloc[tr_idx], label=y_log[tr_idx], categorical_feature=cat_features)
lgb_va_log = lgb.Dataset(
    X_train.iloc[va_idx], label=y_log[va_idx], categorical_feature=cat_features, reference=lgb_tr_log
)

lgb_eval_result_log = {}
lgb_log = lgb.train(
    best_params_log,
    lgb_tr_log,
    num_boost_round=num_round,
    valid_sets=[lgb_va_log, lgb_tr_log],
    valid_names=['valid', 'train'],
    callbacks=[
        lgb.early_stopping(early_stop, first_metric_only=True),
        lgb.record_evaluation(lgb_eval_result_log),
        lgb.log_evaluation(0)
    ]
)

metric_name_log = next(iter(lgb_eval_result_log['valid']))
best_iter = lgb_log.best_iteration or len(lgb_eval_result_log['valid'][metric_name_log])
lgb_log_curve_records = []
tr_features_log = X_train.iloc[tr_idx]
va_features_log = X_train.iloc[va_idx]
tr_target = y_raw[tr_idx]
va_target = y_raw[va_idx]
for iter_idx in range(1, best_iter + 1):
    pred_tr_log = lgb_log.predict(tr_features_log, num_iteration=iter_idx)
    pred_va_log = lgb_log.predict(va_features_log, num_iteration=iter_idx)
    pred_tr = inv(pred_tr_log)
    pred_va = inv(pred_va_log)
    mae_tr = mean_absolute_error(tr_target, pred_tr)
    mae_va = mean_absolute_error(va_target, pred_va)
    lgb_log_curve_records.append((iter_idx, mae_tr, mae_va))

lgb_log_curve = pd.DataFrame(lgb_log_curve_records, columns=['iteration', 'train_mae', 'valid_mae'])
lgb_log_curve_path = CURVE_DIR / 'lgb_log_learning_curve.csv'
lgb_log_curve.to_csv(lgb_log_curve_path, index=False)
print(f"  Learning curve saved: {lgb_log_curve_path}")

pred['LGBM_LOG'] = inv(lgb_log.predict(X_test, num_iteration=lgb_log.best_iteration))
hold['LGBM_LOG'] = inv(lgb_log.predict(X_train.iloc[va_idx], num_iteration=lgb_log.best_iteration))
print(f"  Holdout MAE: {mean_absolute_error(y_raw[va_idx], hold['LGBM_LOG']):.2f}")

# ===================== NNLS Ensemble =====================
print("\n[7] NNLS 앙상블")

# 홀드아웃 구간의 실제 y
y_hold = y_raw[va_idx]

# 모델 이름 리스트
names = list(pred.keys())

# 홀드아웃에서의 각 모델 예측값들을 열 방향으로 쌓아둠 → (n_samples, n_models) 행렬
H_hold = np.column_stack([hold[m] for m in names])
# 테스트셋 예측도 같은 순서로 쌓음
H_test = np.column_stack([pred[m] for m in names])

# NNLS로 "음이 아닌" 가중치를 찾고, 합이 1이 되도록 정규화
w_nnls, _ = SCIPY_NNLS(H_hold, y_hold)
if w_nnls.sum() == 0:
    w_nnls = np.ones_like(w_nnls)
w_nnls /= w_nnls.sum()

# NNLS 가중치로 앙상블한 예측
pred_nnls_test = H_test @ w_nnls
pred_nnls_hold = H_hold @ w_nnls
mae_nnls = mean_absolute_error(y_hold, pred_nnls_hold)

# 개별 모델 성능도 같이 비교해서, NNLS가 더 나쁘면 그냥 제일 좋은 모델만 쓰게끔
ind_mae = {m: mean_absolute_error(y_hold, hold[m]) for m in names}
best_name = min(ind_mae, key=ind_mae.get)
mae_best = ind_mae[best_name]

print("  개별:", {m: round(ind_mae[m], 2) for m in names})
print(f"  NNLS MAE: {mae_nnls:.2f}  | Best: {best_name} {mae_best:.2f}")
print("  가중치:", {names[i]: round(float(w_nnls[i]), 3) for i in range(len(names))})

# 최종으로 쓸 예측값 결정
final_pred = pred_nnls_test if mae_nnls <= mae_best else pred[best_name]
final_label = "NNLS" if mae_nnls <= mae_best else f"Best:{best_name}"
final_hold = min(mae_nnls, mae_best)

# 베스트 파라미터 기록 저장
if best_param_records:
    params_df = pd.DataFrame(best_param_records)
    params_csv_path = CURVE_DIR / 'best_model_params.csv'
    params_df.to_csv(params_csv_path, index=False)
    print(f"  Best params saved: {params_csv_path}")

# ===================== Save =====================
# 제출 파일 형식에 맞춰 id와 예측값을 저장
test_id = pd.read_csv(TEST_PATH)['id'].values
sub = pd.DataFrame({'id': test_id, TARGET: np.clip(final_pred, 0, None)})
sub.to_csv(SUB_PATH, index=False, encoding='utf-8-sig')

print("\n[8] 저장 완료")
print(f"   Path : {SUB_PATH}")
print(f"   Holdout MAE ({final_label}): {final_hold:.2f}")
print("=" * 100)
