# Spoken Language Identification on FLEURS

*From CNN Baselines to XLS-R: Investigating Preprocessing, Speaker Leakage, and Cross-Lingual Representations*

## Overview

This project investigates **spoken language identification (LID)** using a six-language subset of the FLEURS dataset: Mandarin Chinese, Cantonese, Japanese, German, Dutch, and American English.

We developed and evaluated **1D and 2D CNN classifiers using MFCC features**, alongside a pretrained **XLS-R 300M model with a frozen backbone and a lightweight linear classifier**. The experiments cover both six-class classification and binary classification across language pairs.

Our investigation focuses on three aspects:

- **Model architectures:** Comparing 1D and 2D CNNs and evaluating pretrained XLS-R representations.
- **Audio preprocessing:** Examining how input duration, leading-silence trimming, and preprocessing consistency affect classification performance.
- **Generalization:** Investigating speaker leakage and evaluating model performance on unseen speakers.

## Key Findings

- **1D CNN outperformed 2D CNN** in both binary and six-class classification experiments.
- **Longer audio segments generally improved classification performance.** Consistent silence trimming between training and testing achieved substantially better results (90.9% average accuracy) than mismatched preprocessing (77.0%).
- **Speaker leakage substantially inflated the initial CNN results.** Performance dropped sharply under speaker-disjoint evaluation, revealing the models' reliance on speaker-specific cues.
- **Frozen XLS-R representations showed limited effectiveness**, particularly for closely related language pairs such as English–Dutch and German–Dutch. In some binary experiments, the linear classifier collapsed to predicting a single class.

## Conclusion

The project highlights how **data splitting, preprocessing consistency, and representation choices** can strongly influence spoken language identification performance. Our findings emphasize the importance of speaker-disjoint evaluation and careful experimental design when assessing model generalization.
