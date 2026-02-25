# 🚇 Subway Escalator Anomaly Detection & Maintenance Scheduling

지하철역 에스컬레이터의 센서 데이터를 시뮬레이션하고, 머신러닝을 활용해 이상 징후를 탐지하는 프로젝트입니다. 승객 흐름을 방해하지 않는 최적의 유지보수 스케줄링을 목표로 합니다.

## 📌 Project Goals
* **물리 기반 시뮬레이션**: AnyLogic을 활용한 물리 로직 검증 및 Python을 이용한 대량의 시뮬레이션 데이터 생성.
* **이상 탐지 (Anomaly Detection)**: 전류, 온도, 진동 데이터를 분석하여 고장 전조 증상 파악.
* **유지보수 최적화**: 승객 혼잡도를 고려하여 교통 흐름에 지장을 주지 않는 정비 시간대 제안.

## 🛠 Tech Stack
* **Simulation**: AnyLogic (Physical Logic Verification), Python (Data Generation)
* **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning**: PyTorch (AutoEncoder), Scikit-learn (MinMaxScaler, Isolation Forest)

## 📊 Data Overview (Synthetic Data)
* **Features**: Station, Length, Season, Hour, Minute, PassengerCount, Current(A), Temp(°C), Vibration, Label
* **Scenario**: 잠실, 고속터미널, 용두역의 3주간 1분 단위 데이터 (총 9개 시나리오)
* **Status**: ✅ 데이터 생성 완료 및 물리적 타당성 검증(Thermal Equilibrium 등) 완료.

## 📅 Roadmap / TODO
- [x] AnyLogic을 이용한 센서 데이터 물리 로직 정립
- [x] Python 시뮬레이터 개발 및 3주치 시나리오 데이터 생성
- [x] 데이터 전처리 및 스케일링 로직 구현
- [x] simple threshold 모델 설계
- [x] isolation tree 모델 설계 및 학습
- [x] XGBoost 모델 설계 및 학습
- [ ] AutoEncoder 모델 설계 및 학습
- [ ] 이상 탐지 결과 시각화 및 유지보수 스케줄링 로직 제안

---
*개발 중인 레포지토리입니다. (최종 업데이트: 2026-02-25)*
