import pandas as pd
import numpy as np
import os

# 1. 환경 설정
output_dir = "./final_data_v5_pro" 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 기초 데이터 로드
file_path = 'C:/Users/Administrator/Downloads/시간대별 승차인원.csv'
df_pax = pd.read_csv(file_path)

stations = ['잠실', '고속터미널', '용두']
lengths = [40, 20, 10]
time_cols = [f'{i}시-{i+1}시 승차' for i in range(24)]

all_data_list = []

print("🚀 V5.2: '불수능 모드' (희귀 고장 및 변별력 강화) 가동 중...")

for station_name in stations:
    row = df_pax[df_pax['역이름'] == station_name]
    pax_pattern = row[time_cols].values.flatten()
    station_weight = 1.5 if station_name == '고속터미널' else 1.0
    
    for l in lengths:
        results = []
        current_t = 22.0
        cum_aging = 0.0
        fault_remaining_time = 0
        
        for h in range(504): # 3주
            for m in range(60):
                hour = h % 24
                week = h // 168
                
                # 계절 설정
                if week == 0: season, amb_t = "spring", 20.0
                elif week == 1: season, amb_t = "summer", 30.0
                else: season, amb_t = "winter", 5.0
                
                # 기계실 환경 온도 (외기 + 틈새바람 영향)
                base_temp = amb_t + np.random.uniform(5.0, 8.0) 
                
                next_hour = (hour + 1) % 24
                interp_pax = pax_pattern[hour] + (pax_pattern[next_hour] - pax_pattern[hour]) * (m / 60.0)
                actual_pax = max(0, interp_pax + np.random.normal(0, interp_pax * 0.05))
                is_op = (hour >= 5) or (hour < 1) 
                
                if is_op:
                    standing_pax = actual_pax * 0.6
                    walking_pax = actual_pax * 0.4 
                    
                    static_load = (standing_pax * station_weight) / 200000.0
                    dynamic_load = (walking_pax * station_weight * 1.8) / 200000.0
                    total_effective_load = static_load + dynamic_load
                    
                    imbalance_factor = 1.0 + (min(0.5, (actual_pax / 350.0))) if actual_pax > 10 else 1.0
                    
                    # 1. 에이징 (노후화 가속도)
                    aging_step = 0.000005 + (total_effective_load * imbalance_factor * 0.000018)
                    cum_aging += aging_step
                    aging_impact = (np.exp(min(cum_aging * 0.2, 1.5)) - 1) * 0.5
                    
                    # 2. 센서 데이터 생성
                    impulse_noise = np.random.normal(0, 0.008) if walking_pax > 5 else 0
                    curr_i = (l * 1.0) + (total_effective_load * 45) + (aging_impact * 20) + np.random.normal(0, 0.4)
                    
                    # [수정] 진동 베이스
                    vib_base = 0.02 + (aging_impact * 0.65) + (total_effective_load * 0.06)
                    
                    # [수정] 고장 시 진동폭 하향 (0.15 -> 0.04~0.07 수준으로)
                    # 정상 시 노이즈와 고장 시 진동이 살짝 겹치게 하여 모델의 변별력 요구
                    if fault_remaining_time > 0:
                        vib = vib_base + np.random.uniform(0.04, 0.07) + np.random.normal(0, 0.005)
                    else:
                        vib = vib_base + impulse_noise + np.random.normal(0, 0.002)
                    
                    # 3. 열 역학
                    heat_gain = (curr_i**2) * 0.000015 
                    if fault_remaining_time > 0:
                        heat_gain *= 1.3 # 고장 시 과열도 살짝만 더 발생하게
                    
                    cool_loss = (current_t - base_temp) * 0.01 
                    current_t += heat_gain - cool_loss
                    
                    # [수정] 4. 확률적 라벨링 (빈도 대폭 축소)
                    label = 0
                    if fault_remaining_time > 0:
                        label = 1
                        fault_remaining_time -= 1
                    else:
                        # 발생 확률을 1/10 수준으로 낮춤 (진짜 희귀 데이터화)
                        fault_trigger_prob = (total_effective_load * 0.006) + (aging_impact * 0.003)
                        if np.random.random() < fault_trigger_prob:
                            label = 1
                            fault_remaining_time = np.random.randint(10, 30) # 고장 지속 시간도 단축
                    
                    # 노후화가 극도로 심할 때만 라벨 1 (0.6 -> 0.8)
                    if aging_impact > 0.8:
                        label = 1
                        
                else:
                    curr_i, vib, label = 0.0, 0.0, 0
                    current_t -= (current_t - base_temp) * 0.05
                    fault_remaining_time = 0 
                
                results.append([station_name, l, season, hour, m, round(actual_pax, 1), 
                                round(curr_i, 2), round(current_t, 2), round(vib, 4), label])
        
        temp_df = pd.DataFrame(results, columns=['Station', 'Length', 'Season', 'Hour', 'Minute', 
                                                 'PassengerCount', 'Current', 'Temp', 'Vib', 'Label'])
        all_data_list.append(temp_df)
        print(f"  ✅ {station_name} {l}m '불수능' 시뮬레이션 완료")

# 저장
final_df = pd.concat(all_data_list, ignore_index=True)
final_df.to_csv(os.path.join(output_dir, "escalator_v6_pro.csv"), index=False)
print(f"🚀 난이도가 조절된 데이터가 {output_dir}에 저장되었습니다!")



import joblib  # 피클 저장을 위해 권장되는 라이브러리 (sklearn과 호환성 좋음)
from sklearn.preprocessing import RobustScaler

# 1. 데이터 로드 (v5_pro 데이터를 사용하신다고 가정)
input_file = './final_data_v5_pro/escalator_v6_pro.csv'
if not os.path.exists(input_file):
    print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
else:
    df = pd.read_csv(input_file)
    print("✅ 데이터 로드 완료")

    # 2. Feature Engineering
    print("🛠️ 파생 변수 생성 중...")
    group_cols = ['Station', 'Length']

    # 상대적 지표 생성
    df['Rel_Current'] = df.groupby(group_cols)['Current'].transform(lambda x: x / (x.mean() + 1e-6))
    df['Rel_Vib'] = df.groupby(group_cols)['Vib'].transform(lambda x: x / (x.mean() + 1e-6))
    df['Rel_Temp'] = df.groupby(group_cols)['Temp'].transform(lambda x: x / (x.mean() + 1e-6))

    # 승객 1인당 부하
    df['Load_Per_Pax'] = df['Vib'] / (df['PassengerCount'] + 1)

    # 직전 시간대비 변화량
    df['Current_diff'] = df.groupby(group_cols)['Current'].diff().fillna(0)
    df['Vib_diff'] = df.groupby(group_cols)['Vib'].diff().fillna(0)

    # 3. Robust Scaling 적용
    features = [
        'PassengerCount', 'Current', 'Temp', 'Vib', 
        'Rel_Current', 'Rel_Vib', 'Rel_Temp', 
        'Load_Per_Pax', 'Current_diff', 'Vib_diff'
    ]

    print("⚖️ Robust Scaling 적용 및 스케일러 저장 중...")
    scaler = RobustScaler()
    
    # 스케일링 수행
    df[features] = scaler.fit_transform(df[features])

    # 4. 스케일러를 피클 파일로 저장 (.pkl)
    # 나중에 새로운 데이터를 이 스케일러로 똑같이 변환해야 함
    scaler_path = 'robust_scaler_v5.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"💾 스케일러 저장 완료: {scaler_path}")

    # 5. 결과 저장
    output_filename = 'final_preprocessed_v6.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"\n✨ 전처리 및 피클 생성 완료! '{output_filename}' 파일 저장.")