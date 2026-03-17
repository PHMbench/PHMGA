# WaveFilters Ablation

## Operators

- `signal.wavefilters`
- `signal.wavelet_ricker`
- `signal.wavelet_chirplet`
- `signal.wavelet_laplace`
- `signal.wavelet_morlet`

## Default Comparison Order

1. no WaveFilters transform
2. `signal.wavefilters`
3. `signal.wavelet_ricker`
4. `signal.wavelet_morlet`
5. 其余 family

## Notes

- 统一走 `transform_ops.py`
- 参数只走 operator contract，不单独开 config subtree
- 先做 `torch` 路径，再决定是否扩到主表
