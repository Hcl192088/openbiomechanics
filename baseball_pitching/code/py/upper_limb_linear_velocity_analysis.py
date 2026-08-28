#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上肢線性速度分析
Upper Limb Linear Velocity Analysis

功能：
1. 從landmarks數據計算上肢關節的線性速度
2. 分析上肢線性速度的v-t圖
3. 計算上肢線性速度與球速的相關性
4. 生成散布圖和時序圖
5. 統計分析結果

輸入：
- baseball_pitching/data/full_sig/landmarks.zip
- baseball_pitching/data/metadata.csv
- baseball_pitching/data/poi/poi_metrics.csv

輸出：
- 上肢線性速度v-t圖：baseball_pitching/imgs/upper_limb_velocity_time_series.png
- 上肢線性速度與球速散布圖：baseball_pitching/imgs/upper_limb_velocity_vs_ball_speed_scatter.png
- 統計結果更新到：baseball_pitching/ult_mechanics.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import zipfile
import os
from scipy.signal import savgol_filter

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """載入數據"""
    print("📊 載入數據...")
    
    # 載入landmarks數據
    landmarks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'full_sig', 'landmarks.zip')
    with zipfile.ZipFile(landmarks_path, 'r') as zf:
        landmarks_df = pd.read_csv(zf.open('landmarks.csv'))
    
    # 載入元數據
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'metadata.csv')
    metadata_df = pd.read_csv(metadata_path)

    # 載入專案規範的 FP 時間
    poi_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'poi', 'poi_metrics.csv')
    poi_df = pd.read_csv(poi_path, usecols=['session_pitch', 'fp_poi_time'])
    
    print(f"✅ 數據載入完成")
    print(f"   Landmarks數據: {landmarks_df.shape}")
    print(f"   元數據: {metadata_df.shape}")
    print(f"   POI數據: {poi_df.shape}")

    required_landmark_cols = {
        'session_pitch', 'time', 'BR_time',
        'wrist_jc_x', 'wrist_jc_y', 'wrist_jc_z'
    }
    missing_landmark_cols = sorted(required_landmark_cols - set(landmarks_df.columns))
    if missing_landmark_cols:
        raise ValueError(f"Landmarks缺少必要欄位: {missing_landmark_cols}")
    if metadata_df['session_pitch'].duplicated().any():
        raise ValueError("metadata.csv 的 session_pitch 不是唯一，停止分析以避免錯誤合併")
    if poi_df['session_pitch'].duplicated().any():
        raise ValueError("poi_metrics.csv 的 session_pitch 不是唯一，停止分析以避免錯誤合併")
    
    # 檢查關鍵欄位
    print(f"\n📌 Landmarks數據欄位檢查:")
    print(f"   欄位數: {len(landmarks_df.columns)}")
    print(f"   前20個欄位: {landmarks_df.columns.tolist()[:20]}")
    
    # 檢查是否有上肢相關的位置數據
    upper_limb_cols = [col for col in landmarks_df.columns if any(joint in col.lower() for joint in ['wrist', 'elbow', 'shoulder', 'hand'])]
    print(f"   上肢相關欄位: {upper_limb_cols}")
    
    return landmarks_df, metadata_df, poi_df

def calculate_linear_velocity(positions, times, filter_window=5):
    """
    計算線性速度
    
    參數:
    - positions: 位置數據 (numpy array, shape: [n_points, n_dimensions])，單位為 m
    - times: 每一列位置對應的時間 (numpy array)，單位為 s
    - filter_window: 平滑濾波窗口大小
    """
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 1:
        raise ValueError(f"positions 必須是 [n_points, n_dimensions]，實際為 {positions.shape}")
    if len(times) != len(positions):
        raise ValueError("times 與 positions 的列數不一致")

    dt = np.gradient(times)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("time 必須嚴格遞增且能計算有效 dt")

    # 計算速度 = 位置變化 / 實際時間變化，單位為 m/s
    velocities = np.gradient(positions, axis=0) / dt[:, np.newaxis]
    
    # 計算速度大小
    velocity_magnitude = np.sqrt(np.sum(velocities**2, axis=1))
    
    # 平滑處理
    if filter_window > 1:
        velocity_magnitude = savgol_filter(velocity_magnitude, filter_window, 3)
    
    return velocity_magnitude

def extract_upper_limb_metrics(landmarks_df, metadata_df):
    """提取上肢線性速度指標"""
    print("🔍 計算上肢線性速度指標...")
    
    results = []
    
    # 按投球分組處理
    for session_pitch, group in landmarks_df.groupby('session_pitch'):
        group = group.sort_values('time').reset_index(drop=True)
        pitch_speed = group['pitch_speed_mph'].iloc[0] if 'pitch_speed_mph' in group.columns else None
        
        if pd.isna(pitch_speed):
            continue
        
        # 獲取關鍵時間點
        fp_time = group['fp_poi_time'].iloc[0] if 'fp_poi_time' in group.columns else None
        br_time = group['BR_time'].iloc[0] if 'BR_time' in group.columns else None
        
        if pd.isna(fp_time) or pd.isna(br_time):
            continue
        
        # 檢查可用的上肢位置數據
        # 只使用投球手腕；不可把 glove_wrist 混入同一個三維速度向量
        wrist_cols = ['wrist_jc_x', 'wrist_jc_y', 'wrist_jc_z']
        elbow_cols = [col for col in group.columns if 'elbow' in col.lower() and any(coord in col for coord in ['_x', '_y', '_z'])]
        shoulder_cols = [col for col in group.columns if 'shoulder' in col.lower() and any(coord in col for coord in ['_x', '_y', '_z'])]
        
        print(f"   投球 {session_pitch}:")
        print(f"     手腕位置欄位: {wrist_cols}")
        print(f"     手肘位置欄位: {elbow_cols}")
        print(f"     肩膀位置欄位: {shoulder_cols}")
        
        # 計算手腕線性速度
        if len(wrist_cols) >= 3:
            wrist_positions = group[wrist_cols].values
            wrist_velocity = calculate_linear_velocity(wrist_positions, group['time'].to_numpy())
            
            # 計算指標
            max_wrist_velocity = np.max(wrist_velocity)
            
            # 落地時刻速度
            time_diff = np.abs(group['time'] - fp_time)
            fp_idx = np.argmin(time_diff)
            fp_wrist_velocity = wrist_velocity[fp_idx] if fp_idx < len(wrist_velocity) else np.nan
            
            # 球釋放時刻速度
            time_diff = np.abs(group['time'] - br_time)
            br_idx = np.argmin(time_diff)
            br_wrist_velocity = wrist_velocity[br_idx] if br_idx < len(wrist_velocity) else np.nan
            
        else:
            max_wrist_velocity = fp_wrist_velocity = br_wrist_velocity = np.nan
        
        # 計算手肘線性速度
        if len(elbow_cols) >= 3:
            elbow_positions = group[elbow_cols].values
            elbow_velocity = calculate_linear_velocity(elbow_positions, group['time'].to_numpy())
            
            max_elbow_velocity = np.max(elbow_velocity)
            
            time_diff = np.abs(group['time'] - fp_time)
            fp_idx = np.argmin(time_diff)
            fp_elbow_velocity = elbow_velocity[fp_idx] if fp_idx < len(elbow_velocity) else np.nan
            
            time_diff = np.abs(group['time'] - br_time)
            br_idx = np.argmin(time_diff)
            br_elbow_velocity = elbow_velocity[br_idx] if br_idx < len(elbow_velocity) else np.nan
            
        else:
            max_elbow_velocity = fp_elbow_velocity = br_elbow_velocity = np.nan
        
        # 計算肩膀線性速度
        if len(shoulder_cols) >= 3:
            shoulder_positions = group[shoulder_cols].values
            shoulder_velocity = calculate_linear_velocity(shoulder_positions, group['time'].to_numpy())
            
            max_shoulder_velocity = np.max(shoulder_velocity)
            
            time_diff = np.abs(group['time'] - fp_time)
            fp_idx = np.argmin(time_diff)
            fp_shoulder_velocity = shoulder_velocity[fp_idx] if fp_idx < len(shoulder_velocity) else np.nan
            
            time_diff = np.abs(group['time'] - br_time)
            br_idx = np.argmin(time_diff)
            br_shoulder_velocity = shoulder_velocity[br_idx] if br_idx < len(shoulder_velocity) else np.nan
            
        else:
            max_shoulder_velocity = fp_shoulder_velocity = br_shoulder_velocity = np.nan
        
        results.append({
            'session_pitch': session_pitch,
            'pitch_speed_mph': pitch_speed,
            'max_wrist_velocity': max_wrist_velocity,
            'fp_wrist_velocity': fp_wrist_velocity,
            'br_wrist_velocity': br_wrist_velocity,
            'max_elbow_velocity': max_elbow_velocity,
            'fp_elbow_velocity': fp_elbow_velocity,
            'br_elbow_velocity': br_elbow_velocity,
            'max_shoulder_velocity': max_shoulder_velocity,
            'fp_shoulder_velocity': fp_shoulder_velocity,
            'br_shoulder_velocity': br_shoulder_velocity
        })
    
    return pd.DataFrame(results)

def analyze_correlations(upper_limb_data):
    """分析上肢線性速度與球速的相關性"""
    print("📈 分析上肢線性速度與球速相關性...")
    
    velocity_cols = [col for col in upper_limb_data.columns if 'velocity' in col]
    correlations = []
    
    for col in velocity_cols:
        if upper_limb_data[col].notna().sum() < 10:  # 至少需要10個有效數據點
            continue
            
        # 計算相關係數
        valid_data = upper_limb_data[['pitch_speed_mph', col]].dropna()
        
        if len(valid_data) < 10:
            continue
            
        r_pearson, p_pearson = pearsonr(valid_data['pitch_speed_mph'], valid_data[col])
        r_spearman, p_spearman = spearmanr(valid_data['pitch_speed_mph'], valid_data[col])
        
        correlations.append({
            'variable': col,
            'n_samples': len(valid_data),
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'mean_velocity': valid_data[col].mean(),
            'std_velocity': valid_data[col].std()
        })
    
    return pd.DataFrame(correlations)

def create_time_series_plot(landmarks_df, sample_pitches=None):
    """創建上肢線性速度時序圖"""
    print("📊 創建上肢線性速度時序圖...")
    
    if sample_pitches is None:
        # 隨機選擇幾個投球作為樣本
        unique_pitches = landmarks_df['session_pitch'].unique()
        sample_pitches = np.random.choice(unique_pitches, min(5, len(unique_pitches)), replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Upper Limb Linear Velocity Time Series', fontsize=16, fontweight='bold')
    
    # 檢查可用的上肢位置數據
    wrist_cols = ['wrist_jc_x', 'wrist_jc_y', 'wrist_jc_z']
    elbow_cols = [col for col in landmarks_df.columns if 'elbow' in col.lower() and any(coord in col for coord in ['_x', '_y', '_z'])]
    
    for i, session_pitch in enumerate(sample_pitches):
        group = landmarks_df[landmarks_df['session_pitch'] == session_pitch].copy()
        group = group.sort_values('time').reset_index(drop=True)
        
        if len(group) == 0:
            continue
        
        pitch_speed = group['pitch_speed_mph'].iloc[0] if 'pitch_speed_mph' in group.columns else 'Unknown'
        
        # 標準化時間 (相對於落地時間)
        fp_time = group['fp_poi_time'].iloc[0]
        group['time_normalized'] = group['time'] - fp_time
        
        # 計算手腕線性速度
        if len(wrist_cols) >= 3:
            wrist_positions = group[wrist_cols].values
            wrist_velocity = calculate_linear_velocity(wrist_positions, group['time'].to_numpy())
            
            axes[0, 0].plot(group['time_normalized'], wrist_velocity, 
                           alpha=0.7, label=f'Pitch {session_pitch} ({pitch_speed:.1f} mph)')
        
        # 計算手肘線性速度
        if len(elbow_cols) >= 3:
            elbow_positions = group[elbow_cols].values
            elbow_velocity = calculate_linear_velocity(elbow_positions, group['time'].to_numpy())
            
            axes[0, 1].plot(group['time_normalized'], elbow_velocity, 
                           alpha=0.7, label=f'Pitch {session_pitch} ({pitch_speed:.1f} mph)')
    
    # 設置子圖標題和標籤
    axes[0, 0].set_title('Wrist Linear Velocity', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[0, 1].set_title('Elbow Linear Velocity', fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 隱藏未使用的子圖
    axes[1, 0].set_visible(False)
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    
    # 保存圖片
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'imgs', 'upper_limb_velocity_time_series.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 時序圖已保存: {output_path}")
    
    plt.show()

def create_scatter_plots(upper_limb_data, correlations):
    """創建上肢線性速度與球速散布圖"""
    print("📊 創建上肢線性速度與球速散布圖...")
    
    # 選擇有顯著相關性的變數
    significant_correlations = correlations[
        (correlations['pearson_p'] < 0.05) & 
        (correlations['n_samples'] >= 20)
    ].sort_values('pearson_r', key=abs, ascending=False)
    
    if len(significant_correlations) == 0:
        print("⚠️ 沒有找到顯著相關的變數")
        return
    
    # 創建子圖
    n_plots = min(4, len(significant_correlations))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Upper Limb Linear Velocity vs Ball Speed', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(significant_correlations.head(n_plots).iterrows()):
        col = row['variable']
        
        # 過濾有效數據
        valid_data = upper_limb_data[['pitch_speed_mph', col]].dropna()
        
        if len(valid_data) < 10:
            continue
        
        # 創建散布圖
        axes[i].scatter(valid_data['pitch_speed_mph'], valid_data[col], 
                       alpha=0.6, s=30, color='steelblue')
        
        # 添加回歸線
        z = np.polyfit(valid_data['pitch_speed_mph'], valid_data[col], 1)
        p = np.poly1d(z)
        axes[i].plot(valid_data['pitch_speed_mph'], p(valid_data['pitch_speed_mph']), 
                    "r--", alpha=0.8, linewidth=2)
        
        # 設置標題和標籤
        r_val = row['pearson_r']
        p_val = row['pearson_p']
        n_val = row['n_samples']
        
        axes[i].set_title(f'{col.replace("_", " ").title()}\n'
                         f'r = {r_val:.3f}, p = {p_val:.3f}, n = {n_val}', 
                         fontweight='bold')
        axes[i].set_xlabel('Ball Speed (mph)')
        axes[i].set_ylabel('Linear Velocity (m/s)')
        axes[i].grid(True, alpha=0.3)
        
        # 添加統計信息
        axes[i].text(0.05, 0.95, f'Mean: {valid_data[col].mean():.2f} m/s\n'
                                f'Std: {valid_data[col].std():.2f} m/s', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 隱藏未使用的子圖
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存圖片
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'imgs', 'upper_limb_velocity_vs_ball_speed_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 散布圖已保存: {output_path}")
    
    plt.show()

def update_mechanics_report(correlations):
    """更新投球機制報告"""
    print("📝 更新投球機制報告...")
    
    # 選擇最強的相關性
    strongest_corr = correlations.loc[correlations['pearson_r'].abs().idxmax()]
    
    def format_p_value(p_value):
        return '< 0.001' if p_value < 0.001 else f'= {p_value:.3f}'

    report_text = f"""## 上肢線性速度與球速關係分析

### 數據概述
- 分析變數數量: {len(correlations)}
- 有效樣本數: {correlations['n_samples'].max()}
- 速度定義: 位置對實際時間的導數，單位為 m/s
- 手腕欄位: `wrist_jc_x/y/z`（僅投球手腕）
- 事件時間: FP 使用 `fp_poi_time`；BR 使用 `BR_time`

### 主要發現
最強相關性: {strongest_corr['variable']} 與球速呈{'正' if strongest_corr['pearson_r'] > 0 else '負'}相關 (r={strongest_corr['pearson_r']:.3f}, p{format_p_value(strongest_corr['pearson_p'])}, n={strongest_corr['n_samples']})

### 詳細相關性分析
"""

    for _, row in correlations.sort_values('pearson_r', key=lambda values: values.abs(), ascending=False).iterrows():
        report_text += (
            f"- {row['variable']}: r={row['pearson_r']:.3f}, "
            f"p{format_p_value(row['pearson_p'])}, n={row['n_samples']}\n"
        )

    report_text += f"""
### 生物力學意義
上肢線性速度反映了投球過程中手臂運動的整體效率；本次結果的絕對速度值已使用實際取樣時間換算。

分析完成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 讀取現有報告
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ult_mechanics.md')
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    except FileNotFoundError:
        existing_content = ""
    
    # 替換既有同名段落，避免重跑後保留舊的速度單位或事件定義
    section_header = '## 上肢線性速度與球速關係分析'
    if section_header in existing_content:
        section_start = existing_content.index(section_header)
        next_section_start = existing_content.find('\n## ', section_start + len(section_header))
        if next_section_start == -1:
            updated_content = existing_content[:section_start] + report_text.rstrip() + '\n'
        else:
            updated_content = (
                existing_content[:section_start]
                + report_text.rstrip()
                + '\n\n'
                + existing_content[next_section_start + 1:]
            )
    else:
        updated_content = existing_content.rstrip() + '\n\n' + report_text.rstrip() + '\n'
    
    # 寫入報告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ 報告已更新: {report_path}")

def main():
    """主函數"""
    print("🚀 開始上肢線性速度分析...")
    
    # 載入數據
    landmarks_df, metadata_df, poi_df = load_data()
    
    # 合併元數據
    landmarks_df = landmarks_df.merge(
        metadata_df[['session_pitch', 'pitch_speed_mph']], 
        on='session_pitch', 
        how='left',
        validate='many_to_one'
    )
    landmarks_df = landmarks_df.merge(
        poi_df[['session_pitch', 'fp_poi_time']],
        on='session_pitch',
        how='left',
        validate='many_to_one'
    )

    if landmarks_df['fp_poi_time'].isna().any():
        raise ValueError("部分投球缺少 fp_poi_time，停止分析，不使用 fp_100_time fallback")
    
    # 提取上肢線性速度指標
    upper_limb_data = extract_upper_limb_metrics(landmarks_df, metadata_df)
    
    if len(upper_limb_data) == 0:
        print("❌ 沒有找到有效的上肢線性速度數據")
        return
    
    print(f"✅ 成功提取 {len(upper_limb_data)} 個投球的上肢線性速度數據")
    
    # 分析相關性
    correlations = analyze_correlations(upper_limb_data)
    
    if len(correlations) == 0:
        print("❌ 沒有找到有效的相關性分析結果")
        return
    
    print(f"✅ 完成 {len(correlations)} 個變數的相關性分析")
    
    # 顯示結果摘要
    print("\n📊 相關性分析結果摘要:")
    for _, row in correlations.iterrows():
        significance = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
        print(f"   {row['variable']}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.3f}, n={row['n_samples']} {significance}")
    
    # 創建視覺化
    create_time_series_plot(landmarks_df)
    create_scatter_plots(upper_limb_data, correlations)
    
    # 更新報告
    update_mechanics_report(correlations)
    
    print("🎉 上肢線性速度分析完成！")

if __name__ == "__main__":
    main()
