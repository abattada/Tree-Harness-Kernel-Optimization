# Ablation comparison

## Best valid speedup (arm x operator)

| arm             |   cross_entropy |   embedding |   fused_linear_cross_entropy |   geglu |   gemm |   int4_gemm |   kl_div |   layer_norm |   rms_norm |   rope |   softmax |   sum |   swiglu |   vector_add |   vector_exp |   welford |
|:----------------|----------------:|------------:|-----------------------------:|--------:|-------:|------------:|---------:|-------------:|-----------:|-------:|----------:|------:|---------:|-------------:|-------------:|----------:|
| agent_int4_gemm |           0     |       0     |                        0     |   0     |  0     |        1.09 |    0     |        0     |      0     |  0     |     0     | 0     |    0     |        0     |        0     |     0     |
| minimal_kb      |           2.182 |       1.001 |                        0     |   1.689 |  0     |        0    |    4.609 |        1.069 |      2.322 |  4.288 |     1.005 | 0.988 |    0     |        0.999 |        1.001 |     2.427 |
| pilot_full_pro  |           2.114 |       1.005 |                        1.212 |   1.678 |  0.937 |        0    |    4.605 |        0     |      2.302 |  4.409 |     0     | 0     |    1.672 |        0     |        0     |     2.168 |

## fast_p / cost per arm

| arm             |   n_operators |   fast_1.0 |   fast_1.2 |   fast_1.5 |   n_candidates |   correct_rate |   gpu_minutes |   mtokens |
|:----------------|--------------:|-----------:|-----------:|-----------:|---------------:|---------------:|--------------:|----------:|
| agent_int4_gemm |             1 |      1     |      0     |      0     |             10 |          0.8   |           1   |      0    |
| minimal_kb      |            14 |      0.714 |      0.429 |      0.429 |            222 |          0.743 |          15.4 |      1.52 |
| pilot_full_pro  |            12 |      0.75  |      0.667 |      0.583 |            138 |          0.5   |           8.8 |      1.37 |
