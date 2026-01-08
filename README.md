# Game-Theoretic-Defenses-for-Adversarially-Robust-Conformal-Prediction
# Game-Theoretic Defenses for Adversarially Robust Conformal Prediction

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=v1slR4T7Ld)
[![Venue](https://img.shields.io/badge/Venue-TMLR%202026-red)](https://jmlr.org/tmlr/)

This repository contains the official implementation of the paper **"Game-Theoretic Defenses for Adversarially Robust Conformal Prediction"**, published in *Transactions on Machine Learning Research (TMLR), 01/2026*.

## 📋 Abstract

Adversarial attacks pose major challenges to the reliability of deep learning models in safety-critical domains. While Conformal Prediction (CP) provides uncertainty quantification, it relies on the exchangeability assumption, which is violated under adversarial attacks.

We propose a synergistic framework that integrates **Conformal Prediction** with **Game-Theoretic Defense Strategies**. By modeling the adversarial interaction as a discrete, zero-sum game between an attacker and a defender, we derive a **Nash Equilibrium defense strategy**. We prove that this strategy maintains valid coverage while minimizing the worst-case prediction set size against optimal adversaries.

**Key Features:**
*   **Game-Theoretic Formulation:** Models the attacker-defender interaction as a zero-sum game.
*   **Nash Equilibrium Strategy:** Provides a mixed strategy for model selection that is robust against worst-case attacks.
*   **Robust Coverage Guarantee:** Theoretically proven to maintain coverage validity under adversarial perturbations.
*   **SOTA Performance:** Demonstrated effectiveness on CIFAR-10, CIFAR-100, and ImageNet using various CP methods (APS, RAPS, TOPK, SAPS, RANK).

