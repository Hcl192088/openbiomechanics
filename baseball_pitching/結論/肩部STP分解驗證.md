---
up: "[[能量轉移與軀幹旋轉]]"
---

# 肩部 STP 分解驗證

## 結論

肩部 STP 的兩側限制端接近平均，但上臂側略常成為較小端。於 `fp_poi_time` 至 BR 的有效 STP transfer 時間點，上臂側較小占53.1%，軀幹側占46.9%；這不支持先前「上臂側占72.4%」的強瓶頸結論。

## 正確公式

`thorax_dist_seg_pwr` 與 `upper_arm_prox_seg_pwr` 是總 segment power（SP），不是純 segment torque power（STP）：

```text
SP = JFP + STP
```

`shoulder_energy_transfer_jfp` 以流入上臂為正，因此：

```text
thorax_stp   = thorax_dist_seg_pwr + shoulder_energy_transfer_jfp
upper_arm_stp = upper_arm_prox_seg_pwr - shoulder_energy_transfer_jfp
```

當兩側 STP 異號時：

```text
transfer magnitude = min(abs(thorax_stp), abs(upper_arm_stp))
```

軀幹側為負、上臂側為正時，transfer 記為正；方向相反時記為負；同號時 torque transfer 為零。

此公式與 full-signal `shoulder_energy_transfer_stp` 的逐幀結果完全對齊：

- 全部有效列：411球
- r = 1.000000000000
- 平均絕對誤差：0.000017 W
- 最大絕對誤差：0.0008 W
- `thorax_stp + upper_arm_stp = shoulder_energy_generated`，平均絕對誤差0.000028 W

## FP 至 BR 結果

- 事件窗：`fp_poi_time` 至 `BR_time`
- 全部時間點：21,205
- 兩側 STP 異號、存在 torque transfer：20,166（95.1%）
- frame-weighted：上臂側較小53.1%，軀幹側較小46.9%
- 各球比例取平均：上臂側53.2%，軀幹側46.8%
- 411球中：226球以上臂側較小時間點居多，174球以軀幹側居多，11球相同
- 依 transfer magnitude 加權：上臂側較小54.6%，軀幹側較小45.4%
- transfer方向：99.3%的有效時間點為軀幹流向上臂

## 限制端分組與球速

以每位投手各球的限制端比例先取平均，再依上臂側較小比例是否高於50%分組：

| 組別 | n | 平均球速 | 平均體重 |
|---|---:|---:|---:|
| 軀幹側較小時間居多 | 41 | 85.82 mph | 93.26 kg |
| 上臂側較小時間居多 | 59 | 84.00 mph | 88.98 kg |

未校正差異為上臂組慢1.82 mph，Welch p = 0.0506；未達預先常用的0.05門檻。由於軀幹組平均重4.28 kg，控制體重後，上臂組慢1.35 mph（p = 0.161，95% CI -3.24至0.55 mph），沒有顯著組別差異。

不切組而使用連續的「上臂側較小時間比例」時，與球速的未校正相關為 r = -0.244（p = 0.014）；控制體重後為 partial r = -0.189（p = 0.059）。因此目前只有「上臂側較常限制者可能稍慢」的方向性訊號，尚未獲得體重校正後的穩健支持。

411球層級、以投手為cluster的敏感度模型同樣未達顯著：控制體重後，上臂限制球慢1.11 mph（p = 0.185，95% CI -2.75至0.53 mph）。主結論仍以100位投手分析為準。

## 錯誤來源

舊算法直接計算：

```text
min(abs(thorax_dist_seg_pwr), abs(upper_arm_prox_seg_pwr))
```

這是對兩側總SP取最小值，混入JFP，不是肩關節兩側STP的分解。舊版72.4%另使用不受信任的 `fp_100_time`；後續以 `fp_poi_time` 重算得到的59.6%仍沿用錯誤SP公式。兩者均不應保留為正式結論。

## 重現

- 日期：2026-07-29
- 腳本：`baseball_pitching/code/py/validate_shoulder_stp_decomposition.py`
- 資料：`baseball_pitching/data/full_sig/energy_flow.csv`
- 公式來源：官方 `baseball_pitching/code/v3d/CMO.v3s` 中，segment power 定義為 JFP 與 STP 相加。
