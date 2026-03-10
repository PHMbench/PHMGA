# 2_25 算子清单与改造台账

## 1. 当前已可用（`rm101_closed_v1` 初始白名单，27 个）

### layer supported (8)
- `detrend`
- `differentiate`
- `fft`
- `filter`
- `hilbert_envelope`
- `integrate`
- `normalize`
- `stft`

### feature supported (19)
- `abs_mean`
- `clearance_factor`
- `crest_factor`
- `entropy`
- `kurtosis`
- `max`
- `mean`
- `min`
- `peak_to_peak`
- `rms`
- `shape_factor`
- `skew`
- `spectral_centroid`
- `spectral_flatness`
- `spectral_kurtosis`
- `spectral_skewness`
- `std`
- `var`
- `zero_crossing_rate`

## 2. 需要处理算子（31 个）

### layer proxy (10)
- `cepstrum`
- `denoise_wavelet`
- `mel_spectrogram`
- `patch`
- `power_to_db`
- `psd`
- `resample`
- `savgol_filter`
- `spectrogram`
- `wavelet_transform`

### layer unsupported (17)
- `arithmetic`
- `coherence`
- `concatenate`
- `convolution`
- `cross_correlation`
- `distance`
- `dtw_distance`
- `element_wise_product`
- `emd`
- `pca`
- `phase_difference`
- `subtract`
- `time_delay_embedding`
- `transfer_function`
- `vmd`
- `vqt`
- `wigner_ville_distribution`

### feature proxy (1)
- `hjorth_parameters`

### feature unsupported (3)
- `approximate_entropy`
- `band_power`
- `permutation_entropy`

## 3. `rm101_closed_v2` 目标补齐（优先级 P0）
1. `band_power`（feature 单频带标量版）
2. `cross_correlation`（受限双输入版）
3. `psd`（layer 原生化）
4. `spectrogram`（layer 原生化）
5. `cepstrum`（layer 原生化）
6. `savgol_filter`（layer 原生化）

## 4. 责任模块
1. 合同与映射：`src/model/explainable/operator_catalog.py`
2. 编译与质量报告：`src/model/explainable/bridge.py`
3. Planner/Execute 合同门禁：`src/agents/plan_agent.py`、`src/agents/execute_agent.py`
4. 训练门禁与工件落盘：`src/agents/deep_model_train_agent.py`
5. 案例配置注入：`scripts/paper/resolve_case_config.py`

