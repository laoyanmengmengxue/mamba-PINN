# Editor Comment 2 COMSOL Long-Horizon Diagnostic

Purpose: provide traceable evidence that independent COMSOL-generated cylinder-wake cases now contain long-duration reference sequences suitable for longer-horizon follow-up tests. These cases are not treated as strict extensions of the public Dou et al. CFD dataset unless an overlap validation is separately passed.

Stable diagnostic window: 20.0--80.0 s.
Candidate observation window: 60.0--65.0 s.
Candidate extrapolation windows: 65.0--67.0 s, 65.0--70.0 s, 65.0--75.0 s.

| Case | time range | dt | mesh points | mean dominant freq. | freq. std | high-frequency energy ratio |
|---|---:|---:|---:|---:|---:|---:|
| single | 0.00--80.00 s | 0.050 s | 4050 | 0.166528 | 0.000000 | 0.006531 |
| three | 0.00--80.00 s | 0.050 s | 5179 | 0.185956 | 0.011444 | 0.008403 |

Interpretation for the response letter:
- The new COMSOL files solve the data-availability issue for constructing longer independent CFD tests.
- They should not be described as a strict continuation of the cited public dataset because the overlap gate is separate.
- A safe response to Editor Comment 2 is to add this as an independent long-duration diagnostic/follow-up benchmark and keep the main public-dataset claim as short-horizon.
- If these data are used for the manuscript, report them separately from the Dou et al. benchmark metrics.

Generated files are stored in this directory with the `editor2_comsol_*` prefix.
