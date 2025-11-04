from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import nnls as SCIPY_NNLS

import lightgbm as lgb
import xgboost as xgb

# ===================== Paths / Const =====================
TRAIN_PATH = './data/train.csv'
TEST_PATH  = './data/test.csv'
SUB_PATH   = 'submission_final.csv' # [수정] 파일명 변경

TARGET   = "전기요금(원)"
DT_COL   = "측정일시"
CAT_COL  = "작업유형"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*100)
print("전기요금 — v1+ (lag48h & samehour_w2 & hour_of_week-cat) + 프로필잔차 + 변화율 + LGBM(log) + NNLS")
print("="*100)

# ===================== Utils =====================
def fix_midnight_rollover(df, dt_col):
    out = df.copy()
    dt = pd.to_datetime(out[dt_col])
    out.loc[(dt.dt.hour==0)&(dt.dt.minute==0), dt_col] = dt.loc[(dt.dt.hour==0)&(dt.dt.minute==0)] + pd.Timedelta(days=1)
    return out

def get_korean_holidays_2024():
    days=set(); a=days.add
    def rng(s,e):
        for d in pd.date_range(s,e,freq="D"): a(d.date())
    a(pd.to_datetime("2024-01-01").date()); rng("2024-02-09","2024-02-12")
    for d in ["2024-03-01","2024-04-10","2024-05-05","2024-05-06","2024-05-15",
              "2024-06-06","2024-08-15","2024-10-03","2024-10-09","2024-12-25"]:
        a(pd.to_datetime(d).date())
    rng("2024-09-17","2024-09-19")
    return days
HOLIDAYS_2024 = get_korean_holidays_2024()

def base_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[DT_COL])
    out["day"]=dt.dt.day; out["hour"]=dt.dt.hour; out["minute"]=dt.dt.minute
    out["weekday"]=dt.dt.weekday; out["is_weekend"]=(out["weekday"]>=5).astype(int)

    ddate=dt.dt.date
    out["is_holiday"]       = [1 if d in HOLIDAYS_2024 else 0 for d in ddate]
    out["is_holiday_eve"]   = [1 if (pd.Timestamp(d)-pd.Timedelta(days=1)).date() in HOLIDAYS_2024 else 0 for d in ddate]
    out["is_holiday_after"] = [1 if (pd.Timestamp(d)+pd.Timedelta(days=1)).date() in HOLIDAYS_2024 else 0 for d in ddate]

    # 주기항(1, 2고조파)
    out["hour_sin"]=np.sin(2*np.pi*out["hour"]/24); out["hour_cos"]=np.cos(2*np.pi*out["hour"]/24)
    out["hour_sin2"]=np.sin(4*np.pi*out["hour"]/24); out["hour_cos2"]=np.cos(4*np.pi*out["hour"]/24)
    out["dow_sin"]=np.sin(2*np.pi*out["weekday"]/7); out["dow_cos"]=np.cos(2*np.pi*out["weekday"]/7)

    out["is_peak_after"]=((out["hour"]>=13)&(out["hour"]<=17)).astype(int)
    out["is_peak_even"]=((out["hour"]>=18)&(out["hour"]<=22)).astype(int)
    out["is_night"]=((out["hour"]>=23)|(out["hour"]<=5)).astype(int)

    out["hour_of_week"] = out["weekday"]*24 + out["hour"]  # 0~167

    # 범주 준비
    if CAT_COL in out.columns:
        out[CAT_COL]=out[CAT_COL].astype('category')
        if 'UNK' not in out[CAT_COL].cat.categories:
            out[CAT_COL]=out[CAT_COL].cat.add_categories(['UNK'])
        out[CAT_COL]=out[CAT_COL].fillna('UNK')
    else:
        out[CAT_COL]=pd.Series(['UNK']*len(out), dtype='category')
    return out

# ----- Step1: OOF 3변수 -----
def oof_3vars(train_f, test_f):
    print("\n[Step-1] OOF 3종 (5fold TS)")
    basic_cols = [
        'hour','day','minute','weekday','is_weekend',
        'hour_sin','hour_cos','hour_sin2','hour_cos2',
        'dow_sin','dow_cos','is_holiday','is_holiday_eve','is_holiday_after',
        'is_peak_after','is_peak_even','is_night','hour_of_week'
    ]
    le = LabelEncoder()
    train_f['cat_encoded'] = le.fit_transform(train_f[CAT_COL])
    test_f['cat_encoded']  = le.transform(test_f[CAT_COL])
    basic_cols += ['cat_encoded']

    Xtr = train_f[basic_cols]; Xte = test_f[basic_cols]
    tscv = TimeSeriesSplit(n_splits=5)

    def run_one(y):
        oof = np.zeros(len(Xtr)); pred=np.zeros(len(Xte))
        for tr_idx, va_idx in tscv.split(Xtr):
            model = lgb.LGBMRegressor(
                objective="regression_l1",
                n_estimators=700, learning_rate=0.05,
                num_leaves=63, subsample=0.9, colsample_bytree=0.9,
                max_depth=-1, min_child_samples=25,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=RANDOM_STATE, n_jobs=-1
            )
            model.fit(Xtr.iloc[tr_idx], y.iloc[tr_idx])
            oof[va_idx]=model.predict(Xtr.iloc[va_idx]); pred+=model.predict(Xte)/5.0
        return oof, pred

    targets = {'전력사용량(kWh)':'oof_kwh','지상무효전력량(kVarh)':'oof_reactive','지상역률(%)':'oof_pf'}
    for raw, newc in targets.items():
        if raw in train_f.columns:
            y = pd.to_numeric(train_f[raw], errors='coerce').fillna(0)
            o, p = run_one(y)
            train_f[newc]=o.clip(min=0); test_f[newc]=p.clip(min=0)
        else:
            train_f[newc]=np.nan; test_f[newc]=np.nan
    return train_f, test_f

# ----- Step2: v1+ 파생 -----
def deriv_from_oof(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(DT_COL)
    step_1h, step_6h, step_24h = 4, 24, 96
    step_48h, step_7d, step_14d = 192, 7*24*4, 14*24*4
    cols = ['oof_kwh','oof_reactive','oof_pf']
    for col in cols:
        s = pd.to_numeric(out[col], errors='coerce')
        out[f'{col}_lag1']   = s.shift(1)
        out[f'{col}_lag1h']  = s.shift(step_1h)
        out[f'{col}_lag6h']  = s.shift(step_6h)
        out[f'{col}_lag24h'] = s.shift(step_24h)
        out[f'{col}_lag48h'] = s.shift(step_48h)      # ✅ 추가
        out[f'{col}_lag7d']  = s.shift(step_7d)

        out[f'{col}_roll6h']   = s.shift(1).rolling(step_6h,  min_periods=3).mean()
        out[f'{col}_roll24h']  = s.shift(1).rolling(step_24h, min_periods=6).mean()
        out[f'{col}_ema24h']   = s.shift(1).ewm(span=step_24h, adjust=False, min_periods=6).mean()

        out[f'{col}_samehour_d1'] = s - s.shift(step_24h)
        out[f'{col}_samehour_w1'] = s - s.shift(step_7d)
        out[f'{col}_samehour_w2'] = s - s.shift(step_14d)   # ✅ 추가
        out[f'{col}_diff1h']      = s.diff(step_1h)

    a = pd.to_numeric(out['oof_kwh'], errors='coerce')
    r = pd.to_numeric(out['oof_reactive'], errors='coerce')
    out['oof_ratio'] = (r/(a+1e-6)).replace([np.inf,-np.inf], np.nan)
    out['oof_pf_proxy'] = a/(a+r+1e-6)
    out['oof_ratio_ema24h'] = out['oof_ratio'].shift(1).ewm(span=step_24h, adjust=False, min_periods=6).mean()
    return out

# ----- Step3: 시간대 프로필 잔차 & 변화율 -----
def add_seasonal_profile_residuals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(DT_COL)
    if "hour_of_week" not in out.columns:
        out["hour_of_week"] = out["weekday"]*24 + out["hour"]
    how = pd.to_numeric(out["hour_of_week"], errors="coerce").fillna(0).astype(int)

    s = pd.to_numeric(out["oof_kwh"], errors="coerce")
    prof = pd.Series(np.nan, index=out.index)

    # 168개 시간대별 과거 평균(누적 평균의 shift)로 프로필 구성 → OOF 방식
    for h in range(168):
        idx = (how == h)
        seq = s.where(idx)
        csum = seq.cumsum()
        ccnt = idx.cumsum()
        prof[idx] = (csum.shift(1) / ccnt.shift(1)).where(ccnt.shift(1) > 0, np.nan)

    out["how_profile_kwh"] = prof
    out["how_resid_kwh"]   = s - prof

    step_7d  = 7*24*4
    step_14d = 14*24*4
    eps = 1e-6
    out["kwh_rate_w1"] = ((s - s.shift(step_7d))  / (np.abs(s.shift(step_7d))  + eps)).clip(-3, 3)
    out["kwh_rate_w2"] = ((s - s.shift(step_14d)) / (np.abs(s.shift(step_14d)) + eps)).clip(-3, 3)
    return out




# ===================== Load =====================
print("\n[1] 데이터 로드...")
train = pd.read_csv(TRAIN_PATH, parse_dates=[DT_COL])
test  = pd.read_csv(TEST_PATH,  parse_dates=[DT_COL])

train = fix_midnight_rollover(train, DT_COL)
test  = fix_midnight_rollover(test, DT_COL)

# 알려진 이상 시점 제거(선택)
filter_time = pd.Timestamp("2024-11-07 00:00:00")
train = train[train[DT_COL] != filter_time]

# 타겟 기준 극단치 제거 (기준을 상위 0.5%로 변경)
if TARGET in train.columns:
    target_series = pd.to_numeric(train[TARGET], errors='coerce')
    # [수정] 0.995 -> 0.993
    q_993 = target_series.quantile(0.993) 
    
    if pd.notna(q_993):
        original_count = train.shape[0]
        # [수정] q_995 -> q_993
        train = train[target_series <= q_993] 
        removed_count = original_count - train.shape[0]
        # [수정] print문 업데이트
        print(f"  타겟 상위 0.7% ( > {q_993:.2f} ) {removed_count}개 행 제거 완료.") 
    else:
         print("  타겟 임계값 계산 실패. 제거를 건너뜁니다.")


train = train.sort_values(DT_COL).reset_index(drop=True)
test  = test.sort_values(DT_COL).reset_index(drop=True)
print(f"  Train: {train.shape}, Test: {test.shape}")

# ===================== Base + OOF + Derivs =====================
print("\n[2] 기본 시간피처 생성")
train_f = base_time_feats(train)
test_f  = base_time_feats(test)

train_f, test_f = oof_3vars(train_f, test_f)

print("\n[3] v1+ 파생")
train_f = deriv_from_oof(train_f)
test_f  = deriv_from_oof(test_f)

print("\n[3-추가] 프로필 잔차 & 변화율")
train_f = add_seasonal_profile_residuals(train_f)
test_f  = add_seasonal_profile_residuals(test_f)

# hour_of_week 카테고리 컬럼
train_f["how_cat"] = train_f["hour_of_week"].astype("string")
test_f["how_cat"]  = test_f["hour_of_week"].astype("string")

# ===================== Features / Missing =====================
print("\n[4] 피처 구성/결측")
raw_power_cols = ['전력사용량(kWh)','지상무효전력량(kVarh)','진상무효전력량(kVarh)','지상역률(%)','탄소배출량(tCO2)']
num_features_train = [
    c for c in train_f.columns
    if (c not in [DT_COL, TARGET, CAT_COL, 'how_cat']) and
       pd.api.types.is_numeric_dtype(train_f[c]) and
       (c not in raw_power_cols)
]
num_features = [c for c in num_features_train if c in test_f.columns]
cat_features = [CAT_COL, 'how_cat']
all_features = num_features + cat_features

# 수치 결측
median_map = {c: pd.to_numeric(train_f[c], errors='coerce').median(skipna=True) for c in num_features}
for c in num_features:
    train_f[c] = pd.to_numeric(train_f[c], errors='coerce').fillna(median_map[c])
    test_f[c]  = pd.to_numeric(test_f[c],  errors='coerce').fillna(median_map[c])

# 카테고리: 문자열 통일 후 합집합으로 카테고리 지정
train_f['how_cat'] = train_f['how_cat'].astype('string')
test_f['how_cat']  = test_f['how_cat'].astype('string')
for col in cat_features:
    train_f[col] = train_f[col].astype('string').fillna('UNK')
    test_f[col]  = test_f[col].astype('string').fillna('UNK')
    cats = pd.Index(pd.unique(pd.concat([train_f[col], test_f[col]], ignore_index=True))).astype(str)
    train_f[col] = pd.Categorical(train_f[col], categories=cats)
    test_f[col]  = pd.Categorical(test_f[col],  categories=cats)

X_train = train_f[all_features].copy()
X_test  = test_f[all_features].copy()

# ===================== Target =====================
# [수정] y_raw, y_log는 이상치 제거 *후*의 train 데이터로 생성되어야 함
y_raw = pd.to_numeric(train[TARGET], errors='coerce').fillna(0).clip(lower=0).values
y_log = np.log1p(y_raw)
inv = lambda p: np.expm1(np.clip(p, None, 20))

# ===================== CV =====================
tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X_train))
tr_idx, va_idx = splits[-1]
print(f"\n[5] 3-Fold TimeSeriesSplit (Holdout: {len(va_idx)})")

# [수정] w(가중치)도 X_train 길이에 맞춰 다시 생성
w = np.ones(len(X_train))
#w[(train_f["is_peak_after"]==1)|(train_f["is_peak_even"]==1)] = 1.25

pred, hold = {}, {}

# ===================== Models =====================

# LightGBM (MAE on raw target)
print("\n[6-1] LightGBM (RAW Target)")
# 1. label을 y_log -> y_raw로 변경
lgb_tr = lgb.Dataset(X_train.iloc[tr_idx], label=y_raw[tr_idx], weight=w[tr_idx], categorical_feature=cat_features)
lgb_va = lgb.Dataset(X_train.iloc[va_idx], label=y_raw[va_idx], weight=w[va_idx], categorical_feature=cat_features)
lgb_model = lgb.train(
    # 2. objective를 'regression_l1' (MAE) 또는 'huber'로 변경
    params=dict(objective='regression_l1', metric='mae',
                # [수정] 과적합 완화
                num_leaves=48, learning_rate=0.035,
                feature_fraction=0.85, bagging_fraction=0.85,
                max_depth=-1, min_child_samples=60,
                reg_alpha=0.15, reg_lambda=0.35,
                random_state=RANDOM_STATE),
    train_set=lgb_tr, num_boost_round=1500,
    valid_sets=[lgb_va], callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)]
)
# 3. inv() 역변환 제거, 키 이름 변경 (LGBM -> LGBM_RAW)
pred['LGBM_RAW'] = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
hold['LGBM_RAW'] = lgb_model.predict(X_train.iloc[va_idx], num_iteration=lgb_model.best_iteration)
# 4. holdout 스코어 확인
print(f"  Holdout MAE (Raw): {mean_absolute_error(y_raw[va_idx], hold['LGBM_RAW']):.2f}")

# XGBoost (raw-log 혼동 방지 위해 log target 학습 → 역변환)
print("\n[6-2] XGBoost")
Xtr = X_train.copy(); Xte = X_test.copy()
for c in cat_features:
    Xtr[c] = Xtr[c].cat.codes
    Xte[c] = Xte[c].cat.codes
xgb_model = xgb.XGBRegressor(
    n_estimators=1500, 
    # [수정] 과적합 완화
    max_depth=8, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.85, 
    reg_lambda=1.2, reg_alpha=0.2,
    random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    early_stopping_rounds=120, objective="reg:squarederror"
)
xgb_model.fit(Xtr.iloc[tr_idx], y_log[tr_idx], sample_weight=w[tr_idx],
              eval_set=[(Xtr.iloc[va_idx], y_log[va_idx])], verbose=False)
pred['XGB'] = inv(xgb_model.predict(Xte))
hold['XGB'] = inv(xgb_model.predict(Xtr.iloc[va_idx]))
print(f"  Holdout MAE: {mean_absolute_error(y_raw[va_idx], hold['XGB']):.2f}")

# HistGradientBoosting (log target)
print("\n[6-3] HGBR")
hgbr = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE, early_stopping=True,
    max_iter=1500, learning_rate=0.04, max_bins=255
)
hgbr.fit(Xtr.iloc[tr_idx], y_log[tr_idx], sample_weight=w[tr_idx])
pred['HGBR'] = inv(hgbr.predict(Xte))
hold['HGBR'] = inv(hgbr.predict(Xtr.iloc[va_idx]))
print(f"  Holdout MAE: {mean_absolute_error(y_raw[va_idx], hold['HGBR']):.2f}")

# 추가: LGBM(Log-target) 별개 한 개 더 (동일 설정)
print("\n[6-4] LightGBM LOG-TARGET (추가)")
lgb_tr_log = lgb.Dataset(X_train.iloc[tr_idx], label=y_log[tr_idx], categorical_feature=cat_features)
lgb_va_log = lgb.Dataset(X_train.iloc[va_idx], label=y_log[va_idx], categorical_feature=cat_features)
lgb_log = lgb.train(
    params=dict(objective='huber', metric='mae',
                # [수정] 과적합 완화
                num_leaves=48, learning_rate=0.035,
                feature_fraction=0.85, bagging_fraction=0.85,
                max_depth=-1, min_child_samples=60,
                reg_alpha=0.15, reg_lambda=0.35,
                random_state=RANDOM_STATE),
    train_set=lgb_tr_log, num_boost_round=1500,
    valid_sets=[lgb_va_log], callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)]
)
pred['LGBM_LOG'] = inv(lgb_log.predict(X_test, num_iteration=lgb_log.best_iteration))
hold['LGBM_LOG'] = inv(lgb_log.predict(X_train.iloc[va_idx], num_iteration=lgb_log.best_iteration))
print(f"  Holdout MAE: {mean_absolute_error(y_raw[va_idx], hold['LGBM_LOG']):.2f}")

# ===================== NNLS Ensemble =====================
print("\n[7] NNLS 앙상블")
y_hold = y_raw[va_idx]
names = list(pred.keys())
H_hold = np.column_stack([hold[m] for m in names])
H_test = np.column_stack([pred[m] for m in names])

w_nnls, _ = SCIPY_NNLS(H_hold, y_hold)
if w_nnls.sum()==0: w_nnls=np.ones_like(w_nnls)
w_nnls /= w_nnls.sum()

pred_nnls_test = H_test @ w_nnls
pred_nnls_hold = H_hold @ w_nnls
mae_nnls = mean_absolute_error(y_hold, pred_nnls_hold)

ind_mae = {m: mean_absolute_error(y_hold, hold[m]) for m in names}
best_name = min(ind_mae, key=ind_mae.get); mae_best = ind_mae[best_name]

print("  개별:", {m: round(ind_mae[m],2) for m in names})
print(f"  NNLS MAE: {mae_nnls:.2f}  | Best: {best_name} {mae_best:.2f}")
print("  가중치:", {names[i]: round(float(w_nnls[i]),3) for i in range(len(names))})

final_pred  = pred_nnls_test if mae_nnls <= mae_best else pred[best_name]
final_label = "NNLS" if mae_nnls <= mae_best else f"Best:{best_name}"
final_hold  = min(mae_nnls, mae_best)

# ===================== Save =====================
test_id = pd.read_csv(TEST_PATH)['id'].values
sub = pd.DataFrame({'id': test_id, TARGET: np.clip(final_pred, 0, None)})
sub.to_csv(SUB_PATH, index=False, encoding='utf-8-sig')

print("\n[8] 저장 완료")
print(f"   Path : {SUB_PATH}")
print(f"   Holdout MAE ({final_label}): {final_hold:.2f}")
print("="*100)

# ==
