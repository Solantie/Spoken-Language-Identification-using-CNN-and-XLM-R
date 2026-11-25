# [Lilium] – Topic 4: Spoken Language Identification

## Team Information

- Member 1: Ran, Li / 3855113 / st200137
- Member 2: Shulin, Li / 3857409 / st200660

## 1. Task Description

- **Task:** The task aims to classify the language of a given short mono-channel speech clip (5–15 seconds) into one of the 6 selected languages labels. It is a multi-class classification problem over a set of 6 diverse languages.
- **Input / Output:** mono-channel speech recordings / language labels
- **Baseline model:** MFCC + CNN
- **Evaluation metric:** macro-average accuracy and macro F1-score

## 2. Dataset

### 2.1 Source
- **Dataset name:** FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
- **Source:** Google / Hugging Face
- **Download:** https://huggingface.co/datasets/google/fleurs

- **Reference:**
  A. Conneau *et al*., “FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech,” *arXiv preprint arXiv:2205.12446*, 2022. [Online]. Available: https://arxiv.org/abs/2205.12446
- **Cite Count:** 622
- **Conference:** 2022 IEEE Spoken Language Technology Workshop
---

### 2.2 Statistics
- **Number of classes:** 6
- **Number of samples:** 6,000 (1,000 per class)
- **Languages covered:** 
  - CJK Group (3): Cantonese (yue_hant_hk), Japanese (ja_jp), Mandarin (cmn_hans_cn)
  - WE Group (3): American English (en_us), German (de_de), Dutch (nl_nl) 
- **Average duration:** 5-15 seconds per utterance 
### 2.3 Splits
- We will sample from the official FLEURS splits to maintain the statistical and demographic characteristics of the original dataset. For each of the 6 languages, we will construct a subset consisting of 800 training samples, 100 validation samples, and 100 testing samples. 


### 2.4 Usage
- The original FLEURS dataset contains 102 languages with approximately 12 hours of audio per language. To ensure feasible training times given our computational constraints, we will construct a selected subset totaling 6,000 samples (1,000 samples per language).
- Our subset is specifically chosen to facilitate an inter- and intra-group linguistic analysis across two distinct language families, namely CJK and WE.
- The full source data size:  
  | Language            | Tag | Data Size |
  |--------------------|--------------|----------|
  | Chinese Mandarin   | cmn_hans_cn  | 2.52 GB  |
  | Cantonese Chinese  | yue_hant_hk  | 1.99 GB  |
  | Japanese           | ja_jp        | 1.96 GB  |
  | German             | de_de        | 2.41 GB  |
  | Dutch              | nl_nl        | 1.67 GB  |
  | American English   | en_us        | 1.84 GB  |
- Our 6,000-sample subset will significantly reduce the required disk space and memory overhead.

## 3. Advanced Direction

- **Skills expecting to gain**:
  1. Mastering the workflow of constructing an audio classification model from scratch.
  2. Gaining practical, industrial-applicable experience in evaluating model robustness and efficiency, bridging the gap between machine learning theory and real-world applications.
  3. Developing methodologies to tackling research bottlenecks. 
  可删掉4. Learning how to convey research ideas through formal academic posters and presentations.


- **Advanced tasks and justification:**

  - **Effect of Input Length on Performance (How long does the recording need to be?)**  

    We will systematically truncate the test audio clips to different lengths (e.g., 500 ms, 1 s, 2 s) and measure how classification accuracy changes. This allows us to investigate why the model fails under limited input. If performance drops at shorter utterance inputs, we will examine whether this is due to insufficient phonetic context (e.g., incomplete words) or lack of acoustic information (e.g., missing prosodic or tonal cues).  

    → *Skill 1, 2, 3*

  - **Noise Robustness and Data Augmentation (How robust is the model to noisy recordings?)**  

    We will introduce artificial background noise at different signal-to-noise ratios (SNR) into the test set and evaluate how model performance changes under noisy conditions.  

    → *Skill 2, 3*

  - **Intra-/inter-Language Family Performance**  

    We will analyze model performance on both intra- and inter-family language classification tasks. In particular, we will examine whether misclassification patterns exist by inspecting predicted labels and identifying which languages are most often confused with each other (e.g., Chinese Mandarin → Cantonese). Both quantitative and qualitative analyses will be conducted to assess whether errors occur more frequently within the same language family or across different families. 

    → *Skill 2, 3*
  
## 4. Timeline

- **Baseline phase:**
  - Dataset work: 2 weeks
    - Confidence level: medium - while we had experience using HuggingFace models, we are less familiar with the datasets library, so creating a manageable subset of the large FLEURS dataset requires additional effort.
  - Baseline classifier: 2 weeks
    - Confidence level: high - having successfully completed a course project of Intro2DL, we are familiar with the engineering pipeline required to train an audio classification model end-to-end.
- **Advanced tasks:** 4 weeks
  - Confidence level: medium - although we have the engineering skills required to build models, a systematic scientific research process required for these advanced tasks is a new to us.

## 5. Risks and Open Questions
- It's our first time constructing an audio classification model from scratch. We are unsure about the performance of the baseline model we choose. 
- We only have limited experiences with academic poster sessions and presentations. Therefore, we are worried our poster does not meet academic standards (We would appreciate it if you could elaborate on how to create an academic poster in later sessions :))
- We struggled with deciding which and how many languages should be included in the training data. 