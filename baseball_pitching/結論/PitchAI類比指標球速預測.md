# PitchAI 類比指標的球速預測力

## 結論

PitchAI 畫面所列指標中，可與 OpenBiomechanics 欄位較可靠對照的 4 項（肩外旋峰值、跨步長、肘內翻力矩峰值、前腳膝伸角速度峰值），對未見投手的球速預測平均絕對誤差為 **3.05 mph（4.91 km/h）**，`R² = 0.30`。再加入定義僅近似的髖肩分離峰值與肩水平外展峰值後，誤差為 **2.83 mph（4.56 km/h）**，`R² = 0.39`。因此這組動作與動力學資料具有中等訊號，但不足以精準預測個人球速。 ^pitchai-analog-velocity

肘力矩對模型貢獻很大：6 指標模型拿掉肘內翻力矩後，MAE 由 2.83 增至 **3.44 mph（5.53 km/h）**，`R²` 由 0.39 降至 0.21。肘力矩是與高速投球同時產生的負荷結果，不能直接解讀成「提高肘力矩就會提高球速」。

## 數字合理性

以 OpenBiomechanics 411 球的分布作尺度檢查：畫面中的肩外旋 171°接近中位數 169.3°；跨步長 99% 身高非常高，但仍在資料範圍內（最大 99.56%）；膝伸角速度 402°/s 合理，略高於中位數 347.8°/s；髖肩分離 45°若與峰值定義相同則偏高，但仍合理（資料最大 50.5°）。

肩水平外展 8°不能直接與本資料的「峰值」中位數 50.7°比較，因畫面可能顯示特定事件角度且座標定義不同。畫面的 peak elbow torque 32 Nm 若直接視為 OpenBiomechanics 的 peak elbow varus moment，會低於本資料最小值 67.9 Nm；在取得 PitchAI 的力矩定義、體節模型與正規化方式前，只能判定為**定義或尺度可能不同**，不能判定量測錯誤。

Peak arm speed 23 m/s、ball path length 172% 與 PK→BR 825 ms 在目前 POI 表沒有確認為同定義的欄位，因此本次未納入模型，也不以 OpenBiomechanics 分布替它們背書。

## 驗證範圍與方法

- 資料：`data/poi/poi_metrics.csv`，411 球、100 位投手；球速 69.5–94.4 mph。
- 驗證：依 `metadata.user` 做 5-fold GroupKFold，同一投手不跨訓練與測試組。
- 模型：ExtraTrees；主要比較 4 個較可靠映射與加入 2 個近似映射的敏感度模型。
- 盲點：這不是 PitchAI 本身的驗證；不同系統的事件、座標、逆動力學與單鏡頭估算誤差可能不一致。Peak arm speed、ball path length、PK→BR 未測試。
- 重現：`code/py/analyze_pitchai_analog_velocity_prediction.py`；完整結果在 `data/pitchai_analog_prediction/report.json`，逐球 out-of-fold 預測在 `oof_predictions.csv`。

狀態：**初步支持（類比欄位的預測能力）；PitchAI 跨系統數值效度仍需驗證。**
