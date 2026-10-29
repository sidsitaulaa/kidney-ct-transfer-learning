# Stage 2: stronger classifiers on the same group-level folds

All pipelines use the group-level CV folds of stage 1 (5 folds, 204 scans, 12,446 images).
Each is tuned on the validation scans of each training fold; the test fold is scored once. "Selected per fold on
validation" picks, in every fold, the stage-2 pipeline with the best validation macro-F1, so its score is the unbiased
estimate for the whole tuning procedure. **Scan level**: the log-probabilities of all slices of a scan are averaged
and the scan receives one label. 95% CIs come from a bootstrap over scans (2,000 resamples).

## Results (sorted by slice-level macro-F1)

| Pipeline | Slice accuracy [95% CI] | Slice macro-F1 [95% CI] | Slice balanced acc. | Slice MCC | Scan accuracy [95% CI] | Scan macro-F1 [95% CI] |
|---|---|---|---|---|---|---|
| Selected per fold on validation (incl. fine-tuned) | 0.8246 [0.7714, 0.8727] | 0.7958 [0.7311, 0.8486] | 0.8227 | 0.7569 | 0.8578 [0.8088, 0.9069] | 0.8493 [0.7922, 0.9014] |
| Fine-tuned ensemble (mean of 2 runs) | 0.8141 [0.7584, 0.8646] | 0.7859 [0.7209, 0.8408] | 0.8184 | 0.7453 | 0.8235 [0.7696, 0.8775] | 0.8133 [0.7506, 0.8700] |
| Fine-tuned MobileNet_224 | 0.8104 [0.7577, 0.8595] | 0.7841 [0.7197, 0.8353] | 0.8214 | 0.7434 | 0.8284 [0.7745, 0.8775] | 0.8175 [0.7529, 0.8724] |
| Fine-tuned MobileNet | 0.7912 [0.7307, 0.8452] | 0.7557 [0.6876, 0.8159] | 0.7819 | 0.7110 | 0.8186 [0.7647, 0.8676] | 0.8107 [0.7490, 0.8629] |
| Ensemble · mean of backbone LogRegs | 0.7230 [0.6589, 0.7830] | 0.6452 [0.5768, 0.7059] | 0.6421 | 0.6009 | 0.7353 [0.6765, 0.7990] | 0.7027 [0.6268, 0.7724] |
| LogReg · MobileNet | 0.6945 [0.6377, 0.7488] | 0.6206 [0.5646, 0.6721] | 0.6189 | 0.5635 | 0.7059 [0.6471, 0.7647] | 0.6853 [0.6111, 0.7520] |
| LogReg · fusion of all backbones | 0.6941 [0.6281, 0.7563] | 0.6156 [0.5545, 0.6748] | 0.6151 | 0.5616 | 0.7353 [0.6765, 0.7942] | 0.7136 [0.6401, 0.7802] |
| Selected per fold on validation | 0.6887 [0.6237, 0.7506] | 0.6105 [0.5463, 0.6687] | 0.6115 | 0.5540 | 0.7500 [0.6912, 0.8088] | 0.7167 [0.6415, 0.7844] |
| Stage 1 MLP · VGG16 | 0.6651 [0.6016, 0.7262] | 0.5901 [0.5282, 0.6438] | 0.5953 | 0.5223 | 0.6814 [0.6176, 0.7451] | 0.6334 [0.5581, 0.7011] |
| LogReg · ResNet50 | 0.6591 [0.5998, 0.7205] | 0.5838 [0.5276, 0.6393] | 0.5868 | 0.5152 | 0.6618 [0.5980, 0.7304] | 0.6207 [0.5443, 0.6943] |
| LogReg · VGG16 | 0.6639 [0.6011, 0.7263] | 0.5786 [0.5169, 0.6335] | 0.5797 | 0.5184 | 0.6912 [0.6275, 0.7549] | 0.6572 [0.5792, 0.7291] |
| Stage 1 MLP · MobileNet | 0.6486 [0.5895, 0.7085] | 0.5692 [0.5116, 0.6200] | 0.5719 | 0.5006 | 0.6520 [0.5882, 0.7157] | 0.6231 [0.5473, 0.6910] |
| LogReg · Xception | 0.6330 [0.5721, 0.6925] | 0.5642 [0.5083, 0.6181] | 0.5681 | 0.4776 | 0.6863 [0.6225, 0.7549] | 0.6404 [0.5690, 0.7126] |
| Stage 1 MLP · InceptionV3 | 0.6265 [0.5709, 0.6823] | 0.5615 [0.5105, 0.6078] | 0.5692 | 0.4711 | 0.6667 [0.6029, 0.7353] | 0.6280 [0.5513, 0.6987] |
| Stage 1 MLP · ResNet50 | 0.6289 [0.5686, 0.6909] | 0.5493 [0.4947, 0.6027] | 0.5551 | 0.4730 | 0.6863 [0.6225, 0.7500] | 0.6639 [0.5891, 0.7302] |
| LogReg · InceptionV3 | 0.6340 [0.5760, 0.6929] | 0.5484 [0.4947, 0.5986] | 0.5475 | 0.4724 | 0.6471 [0.5833, 0.7108] | 0.5960 [0.5225, 0.6651] |
| Stage 1 MLP · Xception | 0.5563 [0.4965, 0.6147] | 0.4968 [0.4430, 0.5482] | 0.5055 | 0.3841 | 0.6225 [0.5588, 0.6912] | 0.5897 [0.5184, 0.6603] |

**Nested-selection estimate (Selected per fold on validation (incl. fine-tuned)):** slice accuracy
0.8246 [0.7714, 0.8727], scan accuracy 0.8578 [0.8088, 0.9069]. Pipeline chosen in each fold:
fold 0: Fine-tuned MobileNet_224; fold 1: Fine-tuned MobileNet_224; fold 2: Fine-tuned MobileNet; fold 3: Fine-tuned MobileNet_224; fold 4: Fine-tuned MobileNet_224.

## Per class

| Pipeline | Cyst slice F1 | Normal slice F1 | Stone slice F1 | Tumor slice F1 | Cyst scan recall | Normal scan recall | Stone scan recall | Tumor scan recall |
|---|---|---|---|---|---|---|---|---|
| Selected per fold on validation (incl. fine-tuned) | 0.885 | 0.867 | 0.686 | 0.745 | 0.913 | 0.875 | 0.825 | 0.750 |
| Fine-tuned ensemble (mean of 2 runs) | 0.861 | 0.866 | 0.659 | 0.757 | 0.884 | 0.854 | 0.778 | 0.708 |
| Fine-tuned MobileNet_224 | 0.912 | 0.827 | 0.648 | 0.750 | 0.913 | 0.771 | 0.825 | 0.708 |
| Fine-tuned MobileNet | 0.800 | 0.879 | 0.627 | 0.717 | 0.841 | 0.875 | 0.778 | 0.750 |
| Ensemble · mean of backbone LogRegs | 0.823 | 0.803 | 0.429 | 0.526 | 0.899 | 0.896 | 0.540 | 0.458 |
| LogReg · MobileNet | 0.789 | 0.785 | 0.356 | 0.552 | 0.841 | 0.854 | 0.508 | 0.542 |
| LogReg · fusion of all backbones | 0.781 | 0.792 | 0.383 | 0.506 | 0.855 | 0.875 | 0.571 | 0.542 |
| Selected per fold on validation | 0.782 | 0.790 | 0.395 | 0.475 | 0.855 | 0.917 | 0.603 | 0.500 |
| Stage 1 MLP · VGG16 | 0.767 | 0.770 | 0.406 | 0.418 | 0.826 | 0.854 | 0.508 | 0.375 |
| LogReg · ResNet50 | 0.733 | 0.778 | 0.357 | 0.467 | 0.812 | 0.812 | 0.492 | 0.375 |
| LogReg · VGG16 | 0.770 | 0.773 | 0.346 | 0.426 | 0.826 | 0.875 | 0.492 | 0.458 |
| Stage 1 MLP · MobileNet | 0.743 | 0.757 | 0.318 | 0.459 | 0.841 | 0.750 | 0.429 | 0.500 |
| LogReg · Xception | 0.742 | 0.724 | 0.370 | 0.420 | 0.870 | 0.708 | 0.571 | 0.417 |
| Stage 1 MLP · InceptionV3 | 0.720 | 0.720 | 0.381 | 0.425 | 0.812 | 0.750 | 0.540 | 0.417 |
| Stage 1 MLP · ResNet50 | 0.716 | 0.755 | 0.342 | 0.385 | 0.797 | 0.833 | 0.540 | 0.458 |
| LogReg · InceptionV3 | 0.724 | 0.741 | 0.321 | 0.407 | 0.797 | 0.833 | 0.460 | 0.333 |
| Stage 1 MLP · Xception | 0.659 | 0.667 | 0.297 | 0.364 | 0.783 | 0.646 | 0.508 | 0.417 |

## Fine-tuning settings tried (chosen on fold-0 validation scans only)

Scores are on the fold-0 validation scans, using the same 1-in-*stride* slices that guided early stopping
(runs with stride 2 share exactly the same validation images).

| Run | Backbone | Slices used | Unfrozen layers | Image size | Fine-tune LR | Fold-0 val macro-F1 (thinned val) | Fold-0 val macro-F1 (full val) | Minutes |
|---|---|---|---|---|---|---|---|---|
| pilotA_MobileNet_s3_u30_lr1e-4 | MobileNet | 1 in 3 | 30 | 192 | 0.0001 | 0.7032 | not recorded | 11 |
| pilotB_MobileNet_s2_u60_lr5e-5 | MobileNet | 1 in 2 | 60 | 192 | 5e-05 | 0.8064 | not recorded | 26 |
| pilotC_EfficientNetB0_s3_u30_lr1e-4 | EfficientNetB0 | 1 in 3 | 30 | 192 | 0.0001 | 0.5940 | not recorded | 10 |
| pilotD_MobileNet_224_s2_u60_lr5e-5 | MobileNet | 1 in 2 | 60 | 224 | 5e-05 | 0.9142 | 0.9120 | 43 |
| pilotE_MobileNetV2_s2_u60_lr5e-5 | MobileNetV2 | 1 in 2 | 60 | 192 | 5e-05 | 0.6259 | 0.6307 | 10 |
| pilotF_MobileNet_224_s2_u60_lr5e-5_e25 | MobileNet | 1 in 2 | 60 | 224 | 5e-05 | 0.9142 | 0.9120 | 47 |
| pilotG_ResNet50_224_s4_u30_lr5e-5 | ResNet50 | 1 in 4 | 30 | 224 | 5e-05 | 0.7092 | 0.7020 | 40 |

## Difference from the best stage-1 model (Stage 1 MLP · VGG16), paired scan-level bootstrap, Holm-corrected

| Pipeline | Δ slice macro-F1 [95% CI] | p (Holm) | Δ scan macro-F1 [95% CI] | p (Holm)  |
|---|---|---|---|---|
| Selected per fold on validation (incl. fine-tuned) | +0.2057 [+0.1424, +0.2689] | 0.008 | +0.2159 [+0.1490, +0.2903] | 0.008 |
| Fine-tuned ensemble (mean of 2 runs) | +0.1958 [+0.1313, +0.2631] | 0.008 | +0.1799 [+0.1116, +0.2547] | 0.008 |
| Fine-tuned MobileNet_224 | +0.1940 [+0.1328, +0.2587] | 0.008 | +0.1841 [+0.1119, +0.2596] | 0.008 |
| Fine-tuned MobileNet | +0.1656 [+0.0974, +0.2334] | 0.008 | +0.1773 [+0.1082, +0.2540] | 0.008 |
| Ensemble · mean of backbone LogRegs | +0.0550 [+0.0193, +0.0921] | 0.055 | +0.0693 [+0.0083, +0.1326] | 0.28 |
| LogReg · MobileNet | +0.0305 [-0.0115, +0.0748] | 1 | +0.0520 [-0.0128, +0.1204] | 1 |
| LogReg · fusion of all backbones | +0.0255 [-0.0135, +0.0677] | 1 | +0.0802 [+0.0234, +0.1403] | 0.108 |
| Selected per fold on validation | +0.0204 [-0.0166, +0.0573] | 1 | +0.0833 [+0.0144, +0.1487] | 0.209 |
| LogReg · ResNet50 | -0.0063 [-0.0511, +0.0406] | 1 | -0.0127 [-0.0895, +0.0665] | 1 |
| LogReg · VGG16 | -0.0115 [-0.0390, +0.0173] | 1 | +0.0239 [-0.0307, +0.0781] | 1 |
| Stage 1 MLP · MobileNet | -0.0209 [-0.0629, +0.0253] | 1 | -0.0103 [-0.0811, +0.0583] | 1 |
| LogReg · Xception | -0.0260 [-0.0703, +0.0217] | 1 | +0.0070 [-0.0707, +0.0909] | 1 |
| Stage 1 MLP · InceptionV3 | -0.0286 [-0.0673, +0.0115] | 1 | -0.0054 [-0.0758, +0.0659] | 1 |
| Stage 1 MLP · ResNet50 | -0.0408 [-0.0810, +0.0017] | 0.6 | +0.0305 [-0.0510, +0.1128] | 1 |
| LogReg · InceptionV3 | -0.0417 [-0.0900, +0.0042] | 0.684 | -0.0374 [-0.1232, +0.0459] | 1 |
| Stage 1 MLP · Xception | -0.0933 [-0.1385, -0.0448] | 0.008 | -0.0437 [-0.1205, +0.0361] | 1 |
