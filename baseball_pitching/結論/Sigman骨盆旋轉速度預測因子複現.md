# Sigman 骨盆旋轉速度預測因子複現

## 結論

OBP 資料可部分複現截圖中的排序趨勢：9 個能由截圖明確定義的指標中，8 個與原表相關方向一致，但不是每個關係都穩定。骨盆在 BR 的旋轉角度、骨盆由 FP 到 BR 的旋轉量、後髖在 BR 的 Y 角度，以及後髖在 FP 的 X 角度，其投手群聚 bootstrap 95% CI 排除 0；其他同向結果仍包含 0。

`Rear Hip Angle Z (FP to BR)` 在 OBP 為正相關，與原表的負相關相反，因此整張表不能視為完整複現。`Hip X+Z` 沒有公式，暫不計算。整體狀態：**部分支持**。

## 讀表時最重要的限制

原作者已說明 `Max Rear Hip Angle X` 是先對 X 角度取絕對值再找最大值。這會混合最大屈曲與最大伸展，不能當成「伸展越多」的直接證據。OBP 的正常角度定義是後髖 X：伸展為正、屈曲為負。

## 結果

| 指標 | OBP r | 投手群聚 bootstrap 95% CI | 原表 r | 方向一致 |
|---|---:|---:|---:|:---:|
| Pelvis Angle Z at Ball Release | +0.329 | [+0.139, +0.493] | +0.268 | 是 |
| Max abs Rear Hip Angle X | +0.045 | [-0.127, +0.202] | +0.257 | 是 |
| Pelvis Angle Z (FP to BR) | +0.411 | [+0.265, +0.537] | +0.251 | 是 |
| Rear Hip Angle Z (FP to BR) | +0.222 | [+0.020, +0.398] | -0.243 | 否 |
| Stride Length | -0.162 | [-0.350, +0.046] | -0.222 | 是 |
| Rear Hip Angle Y at Ball Release | -0.339 | [-0.514, -0.132] | -0.207 | 是 |
| Rear Hip Angle X at Foot Plant | -0.216 | [-0.373, -0.038] | -0.189 | 是 |
| Lead Knee Extension (FP to BR) | +0.154 | [-0.041, +0.332] | +0.188 | 是 |
| Max Rear Hip Angle Y | -0.164 | [-0.370, +0.063] | -0.169 | 是 |

## 方法與資料定義

- 分析單位：每一球，共 411 球。
- 群聚／重抽樣單位：投手，共 100 位；投手群聚 bootstrap 2,000 次。
- 點估計：Pearson `r`。
- 依變項：POI 的 `max_pelvis_rotational_velo`。
- FP：`fp_poi_time`；BR：`BR_time`。
- 事件角度：取離事件時間最近的 360 Hz frame。
- FP 到 BR 變化：`BR 值 - FP 值`。
- `Max abs Rear Hip Angle X`：全訊號 `max(abs(rear_hip_angle_x))`，依原作者回覆重建。
- `Max Rear Hip Angle Y`：全訊號的 signed maximum；截圖沒有說它也取絕對值。
- `Hip X+Z`：截圖沒有定義組合公式，未擅自推測。
- 腳本：`baseball_pitching/code/py/replicate_sigman_pelvis_predictors.py`
- 分析日期：2026-07-29。

群聚 CI 描述的是：若以投手為重抽樣單位，pitch-level Pearson `r` 的不確定範圍。它不是因果效果，也不能把同一投手的多球視為完全獨立證據。
