---
up: "[[機制]]"
---

# 下沉與後腿驅動 (Drop & Rear Leg Drive)

此階段涉及重心下沉與後腿向前的推進力，是動力鏈能量產生的核心。

## 時間節奏

**PKH 到 FP duration（不支持）**: 目前不支持「高球速側只是因為 PKH→FP 時間較短」這個解釋；在現行 `fp_poi_time` 定義下，PKH→FP duration 與球速幾乎無關。全樣本回歸 n = 411, r = -0.010, R² = 0.0001, p = 0.833。詳見 [[骨盆旋轉速度時序分析]]。 ^pkh-fp-duration-speed

## 能量分析
- **質心動能 (CoG Kinetic Energy)**: 
    - 質心速度與球速呈現弱相關 (r = 0.201)。
    - 平均質心速度範圍: 2.27 - 3.58 m/s。
- **質心前向速度**: 反映整體動力鏈的推進效率 (r = 0.201, p < 0.001)。

## 運動學指標
- **後膝屈曲 (Rear Knee Flexion)**: 
    - 球釋放時後膝伸展更充分者球速較快 (r = -0.275)。
    - 這是所有後腿變數中與球速關聯最強的指標。
- **後髖屈曲 (Rear Hip Flexion)**: 
    - 最大後髖屈曲有助於儲存彈性勢能 (r = 0.264)。
    - 快速組 (≥92mph) 呈現明顯的後髖伸展模式 (-2.9°)，與中慢速組不同。
- **後髖屈多、蹲低只是投手間差異，不是可調的 cue（弱支持／同投手內不支持）**（2026-11-22，PKH→`fp_poi_time`）：控制體重後，後髖最大屈曲較大的投手球速略快（投手層級 partial r = 0.23），但同一投手哪幾球屈得多，球速並沒有比較快（p = 0.48）。蹲多低（骨盆、重心下降量除以身高）、後膝屈曲，在投手間都只有 |r| ≤ 0.18，同投手內也無關。證據：腳本 `code/py/analyze_rear_leg_posture_speed.py`，輸出 `rear_leg_posture_speed_outputs/rear_leg_vs_speed.csv`。
- **股骨角度 (Femur Angle)**: 
    - 後腳股骨矢狀面角度（`pelvis_angle_x - rear_hip_angle_x`）和 BR 軀幹前傾在 MER 與 BR 都有強相關；FP-BR 時序 cluster 為 29-100%。這個角度指標不同於 landmark 的 `rear femur posterior x projection ratio`，解讀上較像軀幹前傾姿勢鏈逐步形成，而不是單獨的後股骨投影預測因子；詳見 [[後股骨矢狀面角度時序分析]]。

![下沉與驅動速度比較](../imgs/rear_drive_vs_speed_by_level.png)

## 相關筆記
- [[機制]]
