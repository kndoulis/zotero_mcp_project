# Retrieval Evaluation (Document-Level FAISS Search)

15 hand-written queries evaluated against `search_by_topic` (the Phase 1 document-level FAISS index).

**Mean precision@3: 44.4%**  
**Mean precision@5: 26.7%**  
**Mean hit@1 (gold paper ranked first): 100.0%**

Precision@k is mechanically bounded by `|relevant papers| / k`: for the 11 queries in this set with only one relevant paper, even a perfect rank-1 retrieval caps precision@3 at 33% and precision@5 at 20% - there just aren't 3 or 5 correct papers in the library for that query. hit@1 above is a truer read on retrieval quality here; precision@3/@5 climb accordingly on the multi-answer queries below.

| Query | Relevant paper key(s) | Retrieved top-5 | Hit@1 | P@3 | P@5 |
|---|---|---|---|---|---|
| chest X-ray pneumonia detection with deep learning | KJLL9K6P, X6FKA365 | X6FKA365, KJLL9K6P, UAKN8HW6, IBIH7BK7, UU683AUC | yes | 0.67 | 0.40 |
| generative adversarial networks for medical imaging data augmentation | TM8LQXFS, UAKN8HW6, UU683AUC | TM8LQXFS, UU683AUC, UAKN8HW6, ILK8TNU9, SVPGIR78 | yes | 1.00 | 0.60 |
| spectral normalization for stabilizing GAN training | EKW3W3IJ | EKW3W3IJ, Y8TI7VKY, DG4Y92RS, TM8LQXFS, PI6VZP7Q | yes | 0.33 | 0.20 |
| Wasserstein GAN training stability | DG4Y92RS, Y8TI7VKY | Y8TI7VKY, EKW3W3IJ, DG4Y92RS, ILK8TNU9, TM8LQXFS | yes | 0.67 | 0.40 |
| batch normalization and internal covariate shift | EWILRB4M | EWILRB4M, 8IMGHSWY, EKW3W3IJ, SVPGIR78, PI6VZP7Q | yes | 0.33 | 0.20 |
| residual learning for deep image recognition | IBIH7BK7 | IBIH7BK7, EWILRB4M, JM2AANBZ, ZGN4F2MU, KJLL9K6P | yes | 0.33 | 0.20 |
| long short-term memory recurrent neural networks | HESU4VPU | HESU4VPU, 3V2VQQEY, ZGN4F2MU, JM2AANBZ, MDA5EI9W | yes | 0.33 | 0.20 |
| vanishing and exploding gradients in recurrent neural networks | 3V2VQQEY | 3V2VQQEY, HESU4VPU, MDA5EI9W, ZGN4F2MU, JM2AANBZ | yes | 0.33 | 0.20 |
| Parkinson's disease gait analysis using wearable sensors | 7345VA89, 8VYGUTNU, WZ3LVHK9 | WZ3LVHK9, 7345VA89, 9NF2N9UF, WZPESPK4, 7QD3ELYK | yes | 0.67 | 0.40 |
| explainable AI SHAP feature importance for model predictions | I6GDLBGU | I6GDLBGU, MDA5EI9W, 53H2HF69, SNREZMFH, UWC9FVNK | yes | 0.33 | 0.20 |
| spiking neural network surrogate gradient training | BKRWYUZ4 | BKRWYUZ4, 25RW9TWW, 84C7VU3Y, MGEN63Z8, FF2BK7ZV | yes | 0.33 | 0.20 |
| dropout regularization to prevent neural network overfitting | ZGN4F2MU | ZGN4F2MU, EWILRB4M, 3V2VQQEY, FF2BK7ZV, KJLL9K6P | yes | 0.33 | 0.20 |
| Adam optimizer for stochastic gradient-based optimization | XR6SQPIP | XR6SQPIP, 3V2VQQEY, ZGN4F2MU, EWILRB4M, MDA5EI9W | yes | 0.33 | 0.20 |
| cycle-consistent adversarial networks for unpaired image-to-image translation | PI6VZP7Q | PI6VZP7Q, TM8LQXFS, EKW3W3IJ, UU683AUC, Y8TI7VKY | yes | 0.33 | 0.20 |
| PyTorch deep learning framework design | EPX53IFX | EPX53IFX, EWILRB4M, SNREZMFH, QGYI374R, AMJZUECR | yes | 0.33 | 0.20 |
