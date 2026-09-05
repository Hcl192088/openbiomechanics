# PKH-aligned pelvis rotation velocity SPM

Window: PKH +0 to +260 frames at 360 Hz (0 to 0.722 s).
FP timing uses `fp_poi_time` only. All 411 pitches have a complete window.

## Summary
mode,comparison,n_high,n_low,high_speed_mean,low_speed_mean,high_fp_median_seconds_from_pkh,high_fp_q25_seconds_from_pkh,high_fp_q75_seconds_from_pkh,low_fp_median_seconds_from_pkh,low_fp_q25_seconds_from_pkh,low_fp_q75_seconds_from_pkh,zstar,n_clusters,min_cluster_p,figure
90mph,>90 mph vs <=90 mph,53,358,91.45660377358492,83.70391061452514,0.6889,0.6194,0.7695,0.6278000000000001,0.5721999999999999,0.690925,3.0210855217210604,2,3.1494697787515236e-05,baseball_pitching\imgs\pelvis_velocity_spm_pkh_90mph.png
quartile,top quartile >= 87.9 vs bottom <= 81.4 mph,104,104,90.23076923076923,78.36249999999998,0.6611,0.5930250000000001,0.726375,0.6625000000000001,0.6159250000000001,0.714575,3.042883169665153,0,,baseball_pitching\imgs\pelvis_velocity_spm_pkh_quartile.png
p55_p45,>=55th pct 85.8 vs <=45th pct 84.6 mph,195,186,88.64461538461536,80.49569892473117,0.6305000000000001,0.5861000000000001,0.69165,0.6389,0.581275,0.7028,3.0219481711799334,2,0.004990791999478494,baseball_pitching\imgs\pelvis_velocity_spm_pkh_p55_p45.png

## SPM clusters
mode,start_frame_from_pkh,end_frame_from_pkh,start_seconds_from_pkh,end_seconds_from_pkh,mean_velocity_difference_high_minus_low,cluster_p
90mph,10.498981880135904,55.00159637459938,0.029163838555933068,0.15278221215166493,-23.001598397807527,3.1494697787515236e-05
90mph,219.28354522075534,234.65879689555751,0.6091209589465426,0.6518299913765486,-108.40826753487221,0.02086766175667587
p55_p45,20.73952475572273,33.639043694897246,0.05760979098811869,0.09344178804138124,-13.526902140357576,0.027109172942455517
p55_p45,70.53756224214409,95.50893553478228,0.1959376728948447,0.26530259870772854,-17.438663368734503,0.004990791999478494

## Pitcher-mean sensitivity
Each pitcher is averaged first, then grouped by pitcher mean velocity. This avoids treating repeated pitches as independent subjects.
mode,comparison,n_high_pitchers,n_low_pitchers,n_clusters,min_cluster_p
90mph,>90 mph vs <=90 mph,13,87,0,
quartile,top quartile >= 87.9 vs bottom <= 81.9 mph,25,25,0,
p55_p45,>=55th pct 85.8 vs <=45th pct 84.9 mph,45,45,0,

## Pitcher-mean SPM clusters
No significant clusters.
