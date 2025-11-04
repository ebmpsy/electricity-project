from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

# Optional libs
try:
    import lightgbm as lgb; HAS_LGBM=True
except Exception:
    HAS_LGBM=False
try:
    import xgboost as xgb; HAS_XGB=True
except Exception:
    HAS_XGB=False
try:
    import catboost as cb; HAS_CATBOOST=True
except Exception:
    HAS_CATBOOST=False
try:
    from scipy.optimize import nnls as SCIPY_NNLS; HAS_SCIPY=True
except Exception:
    HAS_SCIPY=False

# ===================== Path / Const =====================
TRAIN_PATH = './data/train.csv'
TEST_PATH  = './data/test.csv'
SUB_PATH   = 'submission_v2_outlier99_drop_timeband.csv'

TARGET   = "전기요금(원)"
DT_COL   = "측정일시"
CAT_COL  = "작업유형"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*100)
print("전기요금 예측 V2 — 이상치삭제(99%) + 시간대분리모델")
print(f"LightGBM={HAS_LGBM}, XGBoost={HAS_XGB}, CatBoost={HAS_CATBOOST}, SciPy(NNLS)={HAS_SCIPY}")
print("="*100)

# ===================== Midnight rollover bug fix =====================
def fix_midnight_rollover(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """HH:MM == 00:00 → 날짜를 +1일 시프트 (id 불변)."""
    out = df.copy()
    dt = pd.to_datetime(out[dt_col])
    mask = (dt.dt.hour == 0) & (dt.dt.minute == 0)
    out.loc[mask, dt_col] = dt.loc[mask] + pd.Timedelta(days=1)
    return out

# ===================== Holidays (2024) =====================
def get_korean_holidays_2024():
    days = set()
    def add(d): days.add(pd.to_datetime(d).date())
    def add_range(s,e):
        for d in pd.date_range(s,e,freq="D"): days.add(d.date())
    add("2024-01-01")
    add_range("2024-02-09","2024-02-12")
    add("2024-03-01"); add("2024-04-10")
    add("2024-05-05"); add("2024-05-06"); add("2024-05-15")
    add("2024-06-06"); add("2024-08-15")
    add_range("2024-09-17","2024-09-19")
    add("2024-10-03"); add("2024-10-09"); add("2024-12-25")
    return days
HOLIDAYS_2024 = get_korean_holidays_2024()

# ===================== Time/Calendar Features =====================
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[DT_COL])

    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["minute"] = dt.dt.minute
    out["weekday"] = dt.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    ddate = dt.dt.date
    out["is_holiday"]       = [1 if d in HOLIDAYS_2024 else 0 for d in ddate]
    out["is_holiday_eve"]   = [1 if (pd.Timestamp(d)-pd.Timedelta(days=1)).date() in HOLIDAYS_2024 else 0 for d in ddate]
    out["is_holiday_after"] = [1 if (pd.Timestamp(d)+pd.Timedelta(days=1)).date() in HOLIDAYS_2024 else 0 for d in ddate]

    verified_peak_hours = [8,9,10,11,13,14,15,16,17,19,20]
    out["is_verified_peak"] = out["hour"].isin(verified_peak_hours).astype(int)
    out["is_peak_morning"]  = ((out["hour"] >= 8) & (out["hour"] <= 10)).astype(int)
    out["is_peak_evening"]  = ((out["hour"] >= 18) & (out["hour"] <= 22)).astype(int)
    out["is_night"]         = ((out["hour"] >= 23) | (out["hour"] <= 5)).astype(int)

    out["hour_sin"]  = np.sin(2*np.pi*out["hour"]/24)
    out["hour_cos"]  = np.cos(2*np.pi*out["hour"]/24)
    out["dow_sin"]   = np.sin(2*np.pi*out["weekday"]/7)
    out["dow_cos"]   = np.cos(2*np.pi*out["weekday"]/7)

    out["weekend_evening"] = out["is_weekend"] * out["is_peak_evening"]
    out["holiday_peak"]    = out["is_holiday"] * out["is_verified_peak"]
    out["weekend_hour"]    = out["is_weekend"] * out["hour"]

    # 범주형 처리
    if CAT_COL in df.columns:
        out[CAT_COL] = df[CAT_COL].astype('category')
        if 'UNK' not in out[CAT_COL].cat.categories:
            out[CAT_COL] = out[CAT_COL].cat.add_categories(['UNK'])
        out[CAT_COL] = out[CAT_COL].fillna('UNK')
    else:
        out[CAT_COL] = pd.Series(['UNK']*len(df), dtype='category')
        if 'UNK' not in out[CAT_COL].cat.categories:
            out[CAT_COL] = out[CAT_COL].cat.add_categories(['UNK'])

    return out

# ===================== NEW: 직전 주차 통계 (시간대별) =====================
def add_weekly_lag_features(df: pd.DataFrame, power_col: str = '전력사용량(kWh)') -> pd.DataFrame:
    """각 시간대별로 직전 7일간의 평균/표준편차 계산"""
    out = df.copy()
    dt = pd.to_datetime(out[DT_COL])
    out['_datetime'] = dt
    out['_hour'] = dt.dt.hour
    
    # 전력사용량이 있는 경우만 (train)
    if power_col in df.columns:
        power = pd.to_numeric(df[power_col], errors='coerce').fillna(0)
        out['_power'] = power
        
        # 시간대별 그룹화하여 rolling 통계 계산
        out = out.sort_values('_datetime').reset_index(drop=True)
        
        # 각 시간대별로 7일 rolling
        lag_mean = []
        lag_std = []
        
        for hour in range(24):
            hour_mask = out['_hour'] == hour
            hour_data = out[hour_mask].copy()
            
            if len(hour_data) > 0:
                # 7일 = 7개 데이터포인트 (같은 시간대)
                hour_data['lag_mean'] = hour_data['_power'].shift(1).rolling(window=7, min_periods=1).mean()
                hour_data['lag_std'] = hour_data['_power'].shift(1).rolling(window=7, min_periods=1).std()
                
                # 원래 인덱스에 맞춰 저장
                for idx, row in hour_data.iterrows():
                    lag_mean.append((idx, row['lag_mean']))
                    lag_std.append((idx, row['lag_std']))
        
        # 결과 매핑
        lag_mean_dict = dict(lag_mean)
        lag_std_dict = dict(lag_std)
        
        out['lag_week_mean_by_hour'] = out.index.map(lambda x: lag_mean_dict.get(x, np.nan))
        out['lag_week_std_by_hour'] = out.index.map(lambda x: lag_std_dict.get(x, np.nan))
        
        out = out.drop(['_power'], axis=1)
    else:
        # test의 경우 NaN으로 초기화 (나중에 median으로 채움)
        out['lag_week_mean_by_hour'] = np.nan
        out['lag_week_std_by_hour'] = np.nan
    
    out = out.drop(['_datetime', '_hour'], axis=1)
    return out

# ===================== OOF-only 파생(누수 방지) =====================
def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """OOF 예측만으로 파생 생성: 무효합/무효비/PF/total/ratio/log"""
    out = df.copy()
    EPS = 1e-6

    kwh_oof        = 'oof_전력사용량(kWh)'
    kvarh_lag_oof  = 'oof_지상무효전력량(kVarh)'
    pf_lag_oof     = 'oof_지상역률(%)'

    kwh_proxy = pd.to_numeric(out.get(kwh_oof, np.nan), errors='coerce')
    muhohap   = pd.to_numeric(out.get(kvarh_lag_oof, np.nan), errors='coerce')
    out['무효합(kVarh)'] = muhohap

    denom = kwh_proxy.replace(0, np.nan)
    out['무효비'] = (out['무효합(kVarh)'] / denom).where(kwh_proxy > EPS, np.nan)
    out['PF(%)']  = pd.to_numeric(out.get(pf_lag_oof, np.nan), errors='coerce')

    out['total_power'] = kwh_proxy + out['무효합(kVarh)']
    out['active_power_ratio'] = (kwh_proxy / out['total_power'].replace(0, np.nan)).where(out['total_power'] > EPS, np.nan)
    out['전력사용량_log'] = np.log1p(kwh_proxy.clip(lower=0))
    return out

# ===================== Load =====================
print("\n[1] 데이터 로드...")
train = pd.read_csv(TRAIN_PATH, parse_dates=[DT_COL])
test  = pd.read_csv(TEST_PATH,  parse_dates=[DT_COL])

# === 00:00 롤오버 교정 (id 그대로) ===
train = fix_midnight_rollover(train, DT_COL)
test  = fix_midnight_rollover(test, DT_COL)

# (선택) 알려진 이상치 시점 제거
filter_time = pd.Timestamp("2024-11-07 00:00:00")
if DT_COL in train.columns:
    train = train[train[DT_COL] != filter_time]

train = train.sort_values(DT_COL).reset_index(drop=True)
test  = test.sort_values(DT_COL).reset_index(drop=True)
print(f"  Train: {train.shape}, Test: {test.shape}")

# ===================== Base Features =====================
print("\n[3] 시간/달력 피처 생성")
train_f = create_time_features(train)
test_f  = create_time_features(test)

# ===================== NEW: 이상치 처리 (99 percentile 삭제) =====================
print("\n[2-delayed] 이상치 처리 (99% 초과 삭제)")
y_raw_original = pd.to_numeric(train[TARGET], errors='coerce').fillna(0).clip(lower=0).values
percentile_99 = np.percentile(y_raw_original, 99)
outlier_mask = y_raw_original <= percentile_99

print(f"  99 percentile: {percentile_99:.0f}")
print(f"  삭제할 샘플: {(~outlier_mask).sum()} / {len(y_raw_original)}")
print(f"  남은 샘플: {outlier_mask.sum()}")

# 이상치 제거
train = train[outlier_mask].reset_index(drop=True)
train_f = train_f[outlier_mask].reset_index(drop=True)

# OOF 보조 인코딩(수치형)
le = LabelEncoder()
train_f['cat_encoded'] = le.fit_transform(train_f[CAT_COL])
test_f['cat_encoded']  = le.transform(test_f[CAT_COL])

basic_features = [
    'hour','day','weekday','is_weekend','minute',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'cat_encoded','is_holiday','is_holiday_eve','is_holiday_after',
    'is_verified_peak','is_peak_morning','is_peak_evening','is_night',
    'weekend_evening','holiday_peak','weekend_hour'
]

# ===================== Step-1: OOF(시간순) — 전력변수 3종 =====================
print("\n[4] 1단계 OOF — 전력사용량/지상무효/지상역률 (n_splits=5)")
power_vars = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '지상역률(%)']
X_basic_tr = train_f[basic_features].copy()
X_basic_te = test_f[basic_features].copy()

def oof_regression_ts(base_estimator_maker, X, y, X_test, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr      = y.iloc[tr_idx]
        est = base_estimator_maker()
        est.fit(Xtr, ytr)
        oof[va_idx] = est.predict(Xva)
        preds += est.predict(X_test) / n_splits
    return oof, preds

def oof_base_factory():
    if HAS_LGBM:
        return lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=800, learning_rate=0.05,
            num_leaves=63, subsample=0.9, colsample_bytree=0.9,
            max_depth=-1, min_child_samples=20,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    return HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, early_stopping=True,
        max_iter=800, learning_rate=0.05, max_bins=255
    )

for pv in power_vars:
    if pv in train.columns:
        ypv = pd.to_numeric(train[pv], errors='coerce').fillna(0)
        oof, te_pred = oof_regression_ts(oof_base_factory, X_basic_tr, ypv, X_basic_te, n_splits=5)
        train_f[f"oof_{pv}"] = np.clip(oof, 0, None)
        test_f[f"oof_{pv}"]  = np.clip(te_pred, 0, None)
    else:
        train_f[f"oof_{pv}"] = np.nan
        test_f[f"oof_{pv}"]  = np.nan

# ===================== Step-2: 파생 (OOF-only) =====================
print("\n[5] 파생 피처 생성 (OOF only)")
train_f = build_derived_features(train_f)
test_f  = build_derived_features(test_f)

# ===================== 중앙값 대체(Train 기준) + 피처셋 =====================
print("\n[6] 결측치 처리 & 피처 구성")
num_features = [
    'day','hour','minute','weekday','is_weekend',
    'is_holiday','is_holiday_eve','is_holiday_after',
    'is_verified_peak','is_peak_morning','is_peak_evening','is_night',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'weekend_evening','holiday_peak','weekend_hour',
    'oof_전력사용량(kWh)', 'oof_지상무효전력량(kVarh)', 'oof_지상역률(%)',
    '무효합(kVarh)', '무효비', 'PF(%)',
    'total_power', 'active_power_ratio', '전력사용량_log'
]
cat_features = [CAT_COL]
all_features = num_features + cat_features

for col in num_features:
    train_f[col] = pd.to_numeric(train_f[col], errors='coerce') if col in train_f.columns else np.nan
    test_f[col]  = pd.to_numeric(test_f[col],  errors='coerce') if col in test_f.columns else np.nan

median_map = {col: train_f[col].median(skipna=True) for col in num_features}
for col in num_features:
    train_f[col] = train_f[col].fillna(median_map[col])
    test_f[col]  = test_f[col].fillna(median_map[col])

# 범주형 일관화
train_f[CAT_COL] = train_f[CAT_COL].astype('category')
if 'UNK' not in train_f[CAT_COL].cat.categories:
    train_f[CAT_COL] = train_f[CAT_COL].cat.add_categories(['UNK'])
train_f[CAT_COL] = train_f[CAT_COL].fillna('UNK')

test_f[CAT_COL] = test_f[CAT_COL].astype('category')
all_cats = pd.Index(sorted(set(train_f[CAT_COL].cat.categories).union(set(test_f[CAT_COL].cat.categories))))
train_f[CAT_COL] = train_f[CAT_COL].cat.set_categories(all_cats)
test_f[CAT_COL]  = test_f[CAT_COL].cat.set_categories(all_cats)
test_f[CAT_COL]  = test_f[CAT_COL].fillna('UNK')

X_train = train_f[all_features].copy()
X_test  = test_f[all_features].copy()
test_id = test['id'].values if 'id' in test.columns else np.arange(len(test))

# ===================== NEW: 시간대별 분리 =====================
print("\n[7] 시간대별 데이터 분리")
# 야간: 23:00-07:59, 주간: 08:00-22:59
night_mask_tr = (X_train['hour'] >= 23) | (X_train['hour'] <= 7)
day_mask_tr = ~night_mask_tr

night_mask_te = (X_test['hour'] >= 23) | (X_test['hour'] <= 7)
day_mask_te = ~night_mask_te

print(f"  야간 Train: {night_mask_tr.sum()}, 주간 Train: {day_mask_tr.sum()}")
print(f"  야간 Test: {night_mask_te.sum()}, 주간 Test: {day_mask_te.sum()}")

# ===================== 타깃: 로그 변환 =====================
y_raw   = pd.to_numeric(train[TARGET], errors='coerce').fillna(0).clip(lower=0).values
y_train = np.log1p(y_raw)
def inv(pred):
    return np.expm1(np.clip(pred, None, 20))

# ===================== 시간대별 모델 학습 함수 =====================
def train_timeband_models(X_tr, y_tr, mask_tr, X_te, mask_te, band_name="Day"):
    """특정 시간대 마스크에 해당하는 데이터로 모델 학습"""
    # 인덱스를 리셋하여 정렬
    X_band_tr = X_tr[mask_tr].reset_index(drop=True)
    y_band_tr = y_tr[mask_tr].reset_index(drop=True)
    X_band_te = X_te[mask_te].reset_index(drop=True)
    
    if len(X_band_tr) == 0:
        print(f"  [{band_name}] 데이터 없음 - 스킵")
        return None, None
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_band_tr))
    last_tr_idx, last_va_idx = splits[-1]
    
    print(f"\n[{band_name}] 모델 학습 (Holdout: {len(last_va_idx)})")
    
    # 피크 가중치
    w_band = np.ones(len(X_band_tr))
    peak_weight = 1.5
    w_band[X_band_tr["is_verified_peak"]==1] = peak_weight
    
    predictions = {}
    holdout_preds = {}
    
    # LightGBM
    if HAS_LGBM:
        print(f"  [{band_name}] LightGBM 학습")
        cat_idx = [i for i, c in enumerate(all_features) if c in cat_features]
        lgb_params = dict(
            objective='huber', metric='mae',
            num_leaves=63, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8,
            max_depth=10, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1
        )
        X_band_lgb = X_band_tr.copy()
        X_te_lgb = X_band_te.copy()
        
        lgb_tr = lgb.Dataset(
            X_band_lgb.iloc[last_tr_idx], label=y_band_tr.iloc[last_tr_idx],
            weight=w_band[last_tr_idx], categorical_feature=cat_idx
        )
        lgb_va = lgb.Dataset(
            X_band_lgb.iloc[last_va_idx], label=y_band_tr.iloc[last_va_idx],
            weight=w_band[last_va_idx], categorical_feature=cat_idx
        )
        lgb_model = lgb.train(
            lgb_params, lgb_tr, num_boost_round=1400,
            valid_sets=[lgb_va],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        p_te = inv(lgb_model.predict(X_te_lgb, num_iteration=lgb_model.best_iteration))
        p_ho = inv(lgb_model.predict(X_band_lgb.iloc[last_va_idx], num_iteration=lgb_model.best_iteration))
        predictions['LightGBM'] = p_te
        holdout_preds['LightGBM'] = p_ho
    
    # XGBoost
    if HAS_XGB:
        print(f"  [{band_name}] XGBoost 학습")
        Xtr = X_band_tr.copy(); Xte = X_band_te.copy()
        for col in cat_features:
            Xtr[col] = Xtr[col].cat.codes
            Xte[col] = Xte[col].cat.codes
        xgb_model = xgb.XGBRegressor(
            n_estimators=1400, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=100, objective="reg:squarederror"
        )
        xgb_model.fit(
            Xtr.iloc[last_tr_idx], y_band_tr.iloc[last_tr_idx],
            sample_weight=w_band[last_tr_idx],
            eval_set=[(Xtr.iloc[last_va_idx], y_band_tr.iloc[last_va_idx])],
            verbose=False
        )
        predictions['XGBoost'] = inv(xgb_model.predict(Xte))
        holdout_preds['XGBoost'] = inv(xgb_model.predict(Xtr.iloc[last_va_idx]))
    
    # CatBoost
    if HAS_CATBOOST:
        print(f"  [{band_name}] CatBoost 학습")
        Xtr_cb = X_band_tr.copy(); Xte_cb = X_band_te.copy()
        for col in cat_features:
            Xtr_cb[col] = Xtr_cb[col].astype('string').fillna('UNK')
            Xte_cb[col] = Xte_cb[col].astype('string').fillna('UNK')
        cat_idx_names = [i for i, c in enumerate(all_features) if c in cat_features]
        cb_model = cb.CatBoostRegressor(
            iterations=1400, learning_rate=0.05, depth=8,
            random_seed=RANDOM_STATE, loss_function='Huber:delta=1.0',
            verbose=False, early_stopping_rounds=100
        )
        cb_model.fit(
            Xtr_cb.iloc[last_tr_idx], y_band_tr.iloc[last_tr_idx],
            cat_features=cat_idx_names, sample_weight=w_band[last_tr_idx],
            eval_set=(Xtr_cb.iloc[last_va_idx], y_band_tr.iloc[last_va_idx])
        )
        predictions['CatBoost'] = inv(cb_model.predict(Xte_cb))
        holdout_preds['CatBoost'] = inv(cb_model.predict(Xtr_cb.iloc[last_va_idx]))
    
    # HistGradientBoosting
    print(f"  [{band_name}] HGBR 학습")
    Xtr = X_band_tr.copy(); Xte = X_band_te.copy()
    for col in cat_features:
        Xtr[col] = Xtr[col].cat.codes
        Xte[col] = Xte[col].cat.codes
    hgbr = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, early_stopping=True,
        max_iter=1400, learning_rate=0.05, max_bins=255
    )
    hgbr.fit(Xtr.iloc[last_tr_idx], y_band_tr.iloc[last_tr_idx], sample_weight=w_band[last_tr_idx])
    predictions['HGBR'] = inv(hgbr.predict(Xte))
    holdout_preds['HGBR'] = inv(hgbr.predict(Xtr.iloc[last_va_idx]))
    
    # 앙상블 - y_raw도 mask에 맞게 필터링
    y_hold_raw = y_raw[mask_tr][last_va_idx]
    model_names = list(predictions.keys())
    
    if len(model_names) == 0:
        return None, None
    
    H_hold = np.column_stack([holdout_preds[m] for m in model_names])
    H_test = np.column_stack([predictions[m] for m in model_names])
    
    # NNLS 앙상블
    if HAS_SCIPY and H_hold.shape[1] > 1:
        w, _ = SCIPY_NNLS(H_hold, y_hold_raw)
        if w.sum() == 0: w = np.ones_like(w)
    else:
        w, *_ = np.linalg.lstsq(H_hold, y_hold_raw, rcond=None)
        w = np.maximum(w, 0)
        if w.sum() == 0: w = np.ones_like(w)
    
    w = w / w.sum()
    pred_nnls_test = H_test @ w
    pred_nnls_hold = H_hold @ w
    mae_nnls = mean_absolute_error(y_hold_raw, pred_nnls_hold)
    
    # 단순평균
    simple_mean_hold = H_hold.mean(axis=1)
    simple_mean_test = H_test.mean(axis=1)
    mae_simple = mean_absolute_error(y_hold_raw, simple_mean_hold)
    
    # 최고 단일 모델
    ind_maes = {m: mean_absolute_error(y_hold_raw, holdout_preds[m]) for m in model_names}
    best_name = min(ind_maes, key=ind_maes.get)
    mae_best = ind_maes[best_name]
    best_test = predictions[best_name]
    
    print(f"  [{band_name}] 개별 Holdout MAE:", {m: round(ind_maes[m],2) for m in model_names})
    print(f"  [{band_name}] NNLS Holdout MAE: {mae_nnls:.2f}")
    print(f"  [{band_name}] Mean Holdout MAE: {mae_simple:.2f}")
    print(f"  [{band_name}] Best Holdout MAE: {mae_best:.2f}")
    print(f"  [{band_name}] NNLS 가중치:", {m: round(float(w[i]),3) for i,m in enumerate(model_names)})
    
    # 최종 선택
    final_pred = pred_nnls_test
    final_method = "NNLS"
    final_mae = mae_nnls
    
    if mae_simple < final_mae:
        final_pred = simple_mean_test
        final_method = "SimpleMean"
        final_mae = mae_simple
    if mae_best < final_mae:
        final_pred = best_test
        final_method = f"BestSingle:{best_name}"
        final_mae = mae_best
    
    final_pred = np.clip(final_pred, 0, None)
    
    print(f"  [{band_name}] 선택된 방법: {final_method} (MAE: {final_mae:.2f})")
    
    return final_pred, final_method

# ===================== 야간/주간 모델 각각 학습 =====================
print("\n[8] 야간 모델 학습 (23:00-07:59)")
night_pred, night_method = train_timeband_models(
    X_train, pd.Series(y_train), night_mask_tr,
    X_test, night_mask_te, band_name="Night"
)

print("\n[9] 주간 모델 학습 (08:00-22:59)")
day_pred, day_method = train_timeband_models(
    X_train, pd.Series(y_train), day_mask_tr,
    X_test, day_mask_te, band_name="Day"
)

# ===================== 최종 예측 결합 =====================
print("\n[10] 시간대별 예측 결합")
final_pred = np.zeros(len(X_test))

if night_pred is not None:
    final_pred[night_mask_te] = night_pred
else:
    print("  경고: 야간 예측 실패 - 0으로 채움")

if day_pred is not None:
    final_pred[day_mask_te] = day_pred
else:
    print("  경고: 주간 예측 실패s- 0으로 채움")

final_pred = np.clip(final_pred, 0, None)

# ===================== 저장 =====================
sub = pd.DataFrame({
    'id': test_id,
    TARGET: final_pred
})
sub.to_csv(SUB_PATH, index=False, encoding='utf-8-sig')

print("\n[11] 저장 완료")
print(f"   Path: {SUB_PATH}")
print(f"   야간 방법: {night_method}")
print(f"   주간 방법: {day_method}")
print(f"   Pred Stats -> min={final_pred.min():.0f}, max={final_pred.max():.0f}, mean={final_pred.mean():.0f}")
print("="*100)
print("✅ 완료: V2 - 이상치삭제(99%) + 시간대분리모델(야간/주간)")
print("="*100)