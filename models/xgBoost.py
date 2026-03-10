import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 전처리된 V6(불수능 버전) 데이터 로드
df_final = pd.read_csv('final_preprocessed_v6.csv') # 생성하신 파일명에 맞게 확인!

# 2. 가동 시간 데이터 필터링
df_op = df_final[(df_final['Hour'] >= 5) | (df_final['Hour'] < 1)].copy()

# 3. 학습 데이터 준비
features = [
    'PassengerCount', 'Current', 'Temp', 'Vib', 
    'Rel_Current', 'Rel_Vib', 'Rel_Temp', 
    'Load_Per_Pax', 'Current_diff', 'Vib_diff'
]
X = df_op[features]
y = df_op['Label']

# 4. 데이터 분할 (Stratify는 생명줄입니다)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 불균형 데이터 가중치 재계산
# v6 데이터 사용으로 고장이 매우 줄었기 때문에 Ratio가 이전(1.34)보다 훨씬 크게 나옴
ratio = (y == 0).sum() / (y == 1).sum()
print(f"⚖️ 현재 데이터의 불균형 비율(Ratio): {ratio:.2f}")

# 6. XGBoost v6.0 모델 생성
# '정밀도-재현율(PR) 곡선'을 최적화하여 억울한 오탐(FP)을 줄이는 설정입니다.
model_xgb = xgb.XGBClassifier(
    n_estimators=500,         # 더 복잡해진 경계선을 찾기 위해 트리 개수 상향
    learning_rate=0.03,       # 더 천천히, 꼼꼼하게 학습
    max_depth=7,              # 변수 간의 복합적인 관계를 파악하기 위해 깊이 상향
    min_child_weight=5,       # 너무 미세한 노이즈에 반응하지 않도록 조절 (과적합 방지)
    scale_pos_weight=ratio * 0.8, # 고장이 너무 희귀해졌으므로 가중치를 살짝 낮춰 Precision 방어
    subsample=0.7,            # 매번 데이터의 70%만 사용하여 학습 (일반화)
    colsample_bytree=0.7,     # 매번 변수의 70%만 사용하여 학습 (특정 변수 쏠림 방지)
    gamma=0.1,                # 트리가 너무 복잡해지는 것에 제동을 걺
    random_state=42,
    eval_metric='aucpr'
)

print("🚀 XGBoost v6.0 학습 시작... (진정한 물리적 인과관계를 학습 중입니다!)")
model_xgb.fit(X_train, y_train)

# 7. 예측 및 평가
y_pred = model_xgb.predict(X_test)

print("\n🏆 [XGBoost v6.0] 최종 평가 결과")
print(classification_report(y_test, y_pred))

# 8. 모델 저장
model_path = 'xgboost_model_v6_final.pkl'
joblib.dump(model_xgb, model_path)
print(f"💾 모델 저장 완료: {model_path}")

# 9. 변수 중요도 시각화
importances = model_xgb.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('🚀 XGBoost V6.0 Feature Importance (Contextual Detection)')
plt.show()