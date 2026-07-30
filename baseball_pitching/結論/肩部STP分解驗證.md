---
up: "[[能量轉移與軀幹旋轉]]"
---

# 肩部 STP 分解驗證

## 結論

肩部 STP 的兩側限制端接近平均，但上臂側略常成為較小端。於 `fp_poi_time` 至 BR 的有效 STP transfer 時間點，上臂側較小占53.1%，軀幹側占46.9%；這不支持先前「上臂側占72.4%」的強瓶頸結論。

這個46.9%不是「軀幹能量占STP的46.9%」。肩部STP transfer只計一次，兩側比例只是判定每個時間點由哪一側較小的力矩功率決定 `min(abs(thorax_stp), abs(upper_arm_stp))`。`thorax_stp = M_shoulder · omega_thorax` 是肩關節在軀幹側的力矩功率，不是軀幹節段的旋轉動能，也不是軀幹旋轉動能下降量。

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

### 軀幹側STP的三軸分量

直接以 `-shoulder_thorax_moment_axis * radians(torso_velo_axis)` 重建軀幹側肩部STP。三軸加總與上述 `thorax_stp` 高度對齊（r = 0.99486，MAE = 12.22 W），確認反作用力矩的負號與功率分解方向。

在 FP 至 BR：

- 以每個時間點絕對功率最大的軸判定：z軸88.96%、x軸10.55%、y軸0.49%
- 以三軸絕對功率總和加權：z軸83.54%、x軸13.49%、y軸2.98%

因此z軸不只是角速度最大，也是軀幹側肩部STP的主導功率分量。不能再把「減速代理只看z軸、漏掉其他軸」列為低相關的主要解釋。真正尚未直接驗證的是軀幹節段旋轉動能下降；目前算到的仍是肩關節軀幹側力矩功率，以及經 `min()` 後的肩部transfer。

但z軸占軀幹側肩部STP的83.5%，仍不代表「軀幹旋轉動能的83.5%成為STP」。前者分母是肩關節軀幹側三軸力矩功率絕對值；後者需要軀幹節段慣量張量與三維角速度計算 `K = 0.5 * omega' * I * omega`。目前full-signal CSV、metadata、動態C3D與model C3D均未直接輸出軀幹慣量，但專案其實包含原始建模檔 `baseball_pitching/code/v3d/model/v6_model_hybrid_lm.mdh`，可配合靜態model C3D在Visual3D重建個別慣量；先前「只有CMO輸出腳本、沒有建模檔」的判斷錯誤，已更正。

### 原始模型如何計算軀幹慣量

OpenBiomechanics的CMZ建立流程先套用 `v6_model_hybrid_lm.mdh`，再執行Filter、Events與 `CMO.v3s`。MDH將軀幹命名為Visual3D預設節段 `RTA`（Thorax/Ab），而 `MASS`、`GEOMETRY`、`IXX`、`IYY`、`IZZ` 均未自訂，因此沿用Visual3D的Dempster節段質量與Hanavan幾何慣量預設值。

原始模型對每位投手使用：

```text
m_RTA = 0.355 * body_mass
d = 0.5 * distance(C7, CLAV)
r = 0.5 * distance(RSHO, LSHO)
L = distance(midpoint(CLAV, C7), midpoint(STRN, T10))
```

其中 `d` 是橢圓柱的前後深度半徑，`r` 是肩寬方向半徑，`L` 是軀幹節段長度。Visual3D的橢圓柱公式為：

```text
Ixx = m_RTA * (3*d^2 + L^2) / 12
Iyy = m_RTA * (3*r^2 + L^2) / 12
Izz = m_RTA * (d^2 + r^2) / 4
```

因此原始逆動力學中的 `I` 不是只由體重決定，也不是直接使用 `mass * height^2`；它同時包含體重與每位投手靜態校正得到的軀幹長度、肩寬和前後深度。現有 `mass * delta_omega_sq` 與 `mass * height^2 * delta_omega_sq` 只能視為粗略尺度代理。若要直接重算軀幹動能下降，應由原始Visual3D模型匯出每位投手的RTA慣量，並讓角速度與慣量張量使用同一節段座標系。

若暫以體重近似 `I_T`，使用 `mass * (omega_peak^2 - omega_BR^2)`，與重建STP的原始相關升至r = 0.376（R² = 0.142）；以 `mass * height^2`作慣量尺度時為r = 0.447（R² = 0.200）。但目標能量本身與體重高度相關，因此主要是共享體型訊號：體重基準模型已解釋STP的32.8%，加入質量加權下降代理後只額外增加3.5%（p = 0.023）；以體重與身高為基準，`mass * height^2`版本也只額外增加3.6%（p = 0.022）。對總肩部轉移，兩種代理的額外R²分別為4.0%與4.4%。所以「直接乘體重」會提高表面相關，但在體型之外仍只有小幅獨立解釋力；它仍不是個別軀幹慣量的實測值。

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

### 瓶頸當下的功率大小

瓶頸時間比例與功率大小分開計算：

| 當下限制端 | 時間點占比 | 條件平均 transfer power | 條件中位 transfer power | 累積功率樣本占比 |
|---|---:|---:|---:|---:|
| 軀幹側 | 46.9% | 1147 W | 1103 W | 45.4% |
| 上臂側 | 53.1% | 1223 W | 1218 W | 54.6% |

上臂限制時的條件平均功率比軀幹限制時高6.6%。以每位投手先平均後配對，上臂限制時為1269 W、軀幹限制時為1104 W（paired p < 0.001）。因此時間占比沒有掩蓋「軀幹限制時功率特別大」；資料反而顯示上臂限制端在時間和功率強度上都略占較多。

將同一位投手的軀幹旋轉動能下降代理 `omega_peak^2 - omega_BR^2` 與兩部分拆開比較，控制體重後：

| STP部分 | partial r | p |
|---|---:|---:|
| 軀幹限制時的條件平均功率 | -0.080 | 0.431 |
| 上臂限制時的條件平均功率 | 0.248 | 0.013 |
| 軀幹限制時間點的累積功率樣本 | -0.195 | 0.052 |
| 上臂限制時間點的累積功率樣本 | 0.475 | <0.001 |
| 兩部分合計 | 0.232 | 0.020 |
| 上臂側較小的時間比例 | 0.269 | 0.007 |

這表示減速代理的正向訊號主要出現在「上臂側較小、由上臂端決定實際transfer」的時間，而不是軀幹側成為較小端的時間；減速越大者也更常以上臂側為較小端。兩部分加總時方向相反，會削弱總STP與減速代理的整體相關。`累積功率樣本`是 transfer power 乘上近乎固定取樣間隔前的等比例量；資料時間因四位小數儲存而在0.0027與0.0028秒間交替，因此此處用於分解占比與相關，不冒充精確事件積分焦耳。

上述百分比的分母不同，不能相乘或直接推導解釋率：z軸83.5%是「軀幹側肩部STP內部」的軸向組成；軀幹限制部分45.4%是「實際transfer累積功率樣本」的組成；約5%則是100位投手之間，控制體重後 `delta_omega_sq` 與總STP的共享變異。高組成占比不保證投手間共變異高。

控制體重後進一步將 `delta_omega_sq` 與總累積功率樣本的共變異拆開：上臂限制部分提供總正向共變異的204.9%，軀幹限制部分貢獻-104.9%，兩者相加才是100%。超過100%不是能量占比，而是因一部分正向、一部分負向；它直接顯示軀幹限制部分抵銷了約一半的上臂限制正向訊號。

### 限制端的時間順序

限制端不是完全雜亂，而有明顯的群體相位結構。FP至BR前60%主要為上臂側較小，之後逐漸轉向軀幹側較小；70–80%時軀幹側占61.8%，80–90%占87.8%，90–100%占93.5%。BR最後一個有效時間點有92.7%的球為軀幹側較小。

逐球原始訊號仍常有早期交錯：切換次數中位數2次（IQR 2–3），76.2%的球至少切換兩次。最常見序列為 `軀幹→上臂→軀幹`（31.4%）、`上臂→軀幹→上臂→軀幹`（27.7%）及`上臂→軀幹`（21.7%）。所以較準確的描述是「早中期可交錯，但末段高度一致地收斂到軀幹側限制」，不是單一固定切換，也不是隨機散布。

功率累積確認時間占比會高估末段軀幹限制的重要性。FP至BR前30%只累積22.9%的transfer power；50–80%單獨貢獻44.6%；至MER已累積88.5%，MER後只剩11.5%。MER前的累積功率中60.3%由上臂側限制，MER後則88.9%由軀幹側限制。換成全部transfer power的四分法，最大單一區塊是「MER前＋上臂限制」53.4%，其次為「MER前＋軀幹限制」35.2%，「MER後＋軀幹限制」10.2%，「MER後＋上臂限制」1.3%。因此能量大宗確實落在MER前的上臂限制區間；末段雖幾乎都由軀幹限制，但剩餘可轉移功率已少。

### 總下降量與STP功率的SPM時序

以100位投手為獨立分析單位，每球FP至BR正規化101點後先在投手內平均。預測量為投手平均 `omega_peak^2 - omega_BR^2`，結果曲線為正向STP transfer power，SPM GLM同時控制體重。總STP沒有顯著cluster；拆開限制端後，只有上臂限制功率在FP–BR 52.74–63.87%出現顯著正相關cluster（cluster p = 0.000047，區間平均partial r = 0.354，峰值partial r = 0.378，位於57%）。即使對總STP、上臂限制、軀幹限制三個SPM檢定作Bonferroni校正，該cluster仍成立（adjusted p約0.00014）。軀幹限制功率沒有顯著cluster。

此分析將「能量增加」操作化為瞬時正向STP功率，而預測量仍是每位投手峰值至BR的整段總下降量。顯著區間早於MER中位77.18%（IQR 74.66–78.83%），也正位於上臂限制比例及整體transfer power都較高的中段。它只能解讀為：最終總下降量較大的投手，在FP–BR約53–64%的上臂限制階段具有更高STP功率；不能解讀為該時間點的瞬時減速與功率同步增加。

![Trunk deceleration proxy and STP timing](../imgs/trunk_deceleration_stp_spm.png)

### 瞬時減速與瞬時STP功率

為直接回答同步關係，另計算每一時間點的有號減速 `-d(abs(torso_velo_z))/dt`（正值為減速、負值為加速）與同一時間點STP功率的投手間partial r，控制體重。以10,000次投手配對置換的全域max-statistic，同時校正101個時間點與總STP、上臂限制、軀幹限制三條曲線。

軀幹峰值轉速位於FP–BR中位39.45%（IQR 34.78–45.69%）。在每球各自的峰值轉速至BR重新正規化後：

| 功率曲線 | 最大絕對partial r | 位置 | 全域校正 |
|---|---:|---:|---|
| 總STP | -0.170 | peak-to-BR 71% | 無顯著區間 |
| 上臂限制STP | 0.271 | peak-to-BR 22% | 無顯著區間 |
| 軀幹限制STP | -0.242 | peak-to-BR 22% | 無顯著區間 |

因此沒有證據顯示真正減速期內「瞬時減速較大」會在同一時間點伴隨更高STP功率。FP–BR未以峰值對齊的分析只在0–2%的總STP與1–10%的軀幹限制功率出現顯著正相關，但這些區間早於典型峰值轉速，且包含FP邊界導數，不應當作峰值後減速傳能的證據。較合理的整理是：整段總下降量帶有投手層級的動作／容量訊號，但瞬時減速與肩部STP並未呈現直接同步耦合。

![Instantaneous trunk deceleration and STP after peak speed](../imgs/postpeak_trunk_deceleration_stp_correlation.png)

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
- 慣量模型：`baseball_pitching/code/v3d/model/v6_model_hybrid_lm.mdh`
- Visual3D文件：[Build CMZs](https://wiki.has-motion.com/doku.php?id=other%3Ainspect3d%3Atutorials%3Abuild_cmzs)、[Segment Mass](https://wiki.has-motion.com/doku.php?id=visual3d%3Adocumentation%3Amodeling%3Asegments%3Asegment_mass)、[Segment Inertia](https://wiki.has-motion.com/doku.php?id=visual3d%3Adocumentation%3Amodeling%3Asegments%3Asegment_inertia)、[Segment Properties Examples](https://www.wiki.has-motion.com/doku.php?id=visual3d%3Adocumentation%3Amodeling%3Asegments%3Asegment_properties_example)
