from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. 전처리된 데이터 로드 (V6 불수능 버전)
df_final = pd.read_csv('final_preprocessed_v6.csv')
df_op = df_final[(df_final['Hour'] >= 5) | (df_final['Hour'] < 1)].copy()

# 2. 학습 데이터 준비 (라벨 없이 피처만 사용!)
features = [
    'PassengerCount', 'Current', 'Temp', 'Vib', 
    'Rel_Current', 'Rel_Vib', 'Rel_Temp', 
    'Load_Per_Pax', 'Current_diff', 'Vib_diff'
]
X = df_op[features]
y_true = df_op['Label'] # 비교를 위한 실제 정답

# 3. Isolation Forest 모델 생성
# contamination: 전체 데이터 중 이상치(고장)가 대략 몇 %인지 설정 (앞서 확인한 6% 정도)
outlier_fraction = (y_true == 1).sum() / len(y_true)

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=outlier_fraction, # 고장 비율 자동 설정
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)

print(f"🚀 Isolation Forest 학습 시작... (예상 이상치 비율: {outlier_fraction:.4f})")
# 피팅 (라벨 y를 넣지 않습니다!)
iso_forest.fit(X)

# 4. 이상치 예측 
# 결과값: 1(정상), -1(이상치) -> 이를 우리 라벨인 0(정상), 1(고장)로 변환
scores = iso_forest.decision_function(X) # 이상치 점수 (낮을수록 위험)
y_pred_raw = iso_forest.predict(X)
y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

# 5. 결과 평가
print("\n🔍 [Isolation Forest] 비지도 학습 기반 이상 탐지 결과")
print(classification_report(y_true, y_pred))

# 6. 이상치 점수 시각화
plt.figure(figsize=(12, 6))
sns.histplot(scores, bins=50, kde=True, color='teal')
plt.axvline(x=np.percentile(scores, outlier_fraction*100), color='red', linestyle='--', label='Anomaly Threshold')
plt.title('Anomaly Score Distribution (Lower is more suspicious)')
plt.legend()
plt.show()

# 모델 저장
joblib.dump(iso_forest, 'iso_forest_v6.pkl')
print("💾 Isolation Forest 모델 저장 완료!")