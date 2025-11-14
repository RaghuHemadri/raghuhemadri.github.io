---
layout: post
title: "ML Sys Design: How Federated Learning Protects Your Privacy While Improving Results"
date: 2025-11-13 09:00:00
description: 
tags: system-design ml
categories: ml
thumbnail: assets/img/fed_learning.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# Building the Invisible Search Engine: How Federated Learning Protects Your Privacy While Improving Results

We are diving deep into how modern search systems can become incredibly smart and personalized—without ever accessing your sensitive personal data. This revolutionary approach, known as **Privacy-Preserving Federated Comparison of Search Behaviour**, allows search engines to learn directly from hundreds of millions of user devices while keeping raw data securely locked down.

---

## **Stage 1: Understanding the Privacy Problem**

Traditionally, to make search better, companies collected massive centralized search logs. These logs captured everything: your queries, clicks, dwell times (how long you looked at a result), and even abandoned sessions. While great for improving ranking models, this poses **serious privacy concerns**, as these logs can unintentionally reveal highly personal details, such as health searches or private interests.

Our objective is to completely redesign this system. We aim to enable **federated learning of retrieval and ranking models** directly on user devices. This means the search engine learns from behavioral patterns, but the raw data—like your actual search queries or interaction history—**never leaves the user’s control**.

The final system needs to operate successfully at the immense scale of hundreds of millions of devices and meet challenging performance targets, such as **Apple-class latency (under 150 milliseconds)** for interactive queries.

**To summarize concisely:** We are designing a **federated, privacy-preserving retrieval and ranking system** that learns from distributed behavioral signals while ensuring no raw personal data is ever centralized or reconstructed.

---

## **Stage 2: High-Level Architectural Design**

The search system is built upon two core, complementary halves: **on-device learning** and **server-side orchestration**.

1.  **On-Device Learning (The Client):** Each user device hosts a lightweight client. This client locally logs ephemeral search interactions, forms private training examples, and periodically updates the global model weights using federated learning.
2.  **Server-Side Orchestration (The Coordinator):** The server coordinator aggregates these updates using **secure aggregation**. This crucial step ensures that the contributions of any individual device cannot be inspected.

### The Two Core Models

The overall system relies on two main models for processing queries:

1.  **The Retrieval Model:** This model typically uses a **two-tower architecture**. The **QueryTower** runs **on-device** and learns to embed user queries and contextual signals into a vector space. The **ItemTower** resides **on the server** and embeds all candidate items (like apps, songs, or books) into the same space. During a live search, the query embedding is matched against an **Approximate Nearest Neighbor (ANN)** index of item embeddings to quickly fetch the top candidates.
2.  **The Ranking Model:** After initial retrieval, a separate **ranking model** reranks the candidates using fine-grained, localized features like past interactions or device context. Critically, **this reranker also resides on the device**, ensuring that personalized signals never leave the user’s secure storage.

### Training Loop and Privacy

The learning process is strictly federated:

*   Each device computes gradient updates based on its private data.
*   These updates are then clipped and intentionally perturbed with **differential privacy noise**.
*   The encrypted updates are sent to the aggregation server, which uses secure aggregation to combine them into a global model update.
*   This updated global model is then redistributed to the devices in the next round.

This secure cycle of local computation, secure aggregation, and global update forms the **backbone of the system**. Governance layers track essential factors like **differential privacy budgets**, model drift, and ensure fairness across diverse device populations.

---

## **Stage 3: Data Considerations**

Since the system is prevented from centralizing raw logs, a core challenge is that **labels must be inferred locally** from user interactions.

### Inferring Local Labels

We turn user behavior into implicit training signals:

*   A click followed by a **long dwell time** is treated as a positive signal.
*   **Quick bounces** or skipped results are treated as negatives.
*   In specific contexts, such as an App Store, a downloaded but unopened app might be a weak positive, while an app deleted after one use would be a **strong negative**.

These implicit labels are valuable, noisy, and are completely privacy-respecting because **they stay on-device**.

### Features and Preprocessing

The model uses a mixture of features:

*   **Query Features:** Tokenized words, language, and intent classification (informational, transactional, etc.).
*   **Contextual Features:** Device locale, time of day, or interaction modality.
*   **Item Features:** Semantic embeddings, categorical metadata, and global popularity (maintained on the server).
*   **Interaction Features:** Personalized signals like local Click-Through-Rate (CTR) estimates or topic affinities, which are **computed exclusively on-device**.

Preprocessing is also distributed:

*   Text normalization and embedding lookups happen locally.
*   Continuous features are standardized using local statistics, as global statistics are unavailable.
*   To manage missing or faulty data, we use masking indicators and robust estimators.
*   Given that positive interactions (clicks) are often rare—meaning the data is imbalanced—we can apply techniques like **weighted losses or focal loss** to emphasize those crucial positive events.

Finally, evaluation data must respect temporal order to avoid data leakage, and **federated validation** is achieved by evaluating on hold-out device cohorts rather than using centralized datasets, ensuring privacy compliance while maintaining statistical soundness.

---

## **Stage 4: Modeling, Metrics, and Training**

The technical challenge is conceptually separated into two learning tasks:

1.  **Retrieval:** A metric-learning problem aiming to position semantically or behaviorally related query-item pairs closer together in the embedding space.
2.  **Ranking:** A listwise prediction task focused on ordering candidates based on expected user satisfaction or engagement.

### Model Implementation

For retrieval, the **two-tower neural model** is ideal. It is trained using a **contrastive objective** like InfoNCE, which ensures matching pairs have high similarity while pushing non-matching (negative) samples away. Since item data is globally shared, the Item Tower can be trained centrally, but the Query Tower **adapts continuously using federated updates from devices**.

For ranking, a **lightweight neural reranker** runs on the device. This might be a compact MLP or transformer designed to refine the top-k candidates retrieved. Due to limited on-device resources, this model must be highly efficient, typically inferring results within **tens of milliseconds**. Federated learning allows training of shared layers while often fine-tuning small adapter layers specific to each device.

### Training Stability

Training in federated environments is challenging because data distributions across devices are often non-IID (not independently and identically distributed). We manage this instability using advanced algorithms such as **FedProx or adaptive optimizers like FedAdam**. Furthermore, secure aggregation techniques and robust estimators, such as the coordinate-wise median, are used to protect the global model against potentially corrupted or adversarial clients.

### Evaluation and Metrics

Evaluation proceeds in two crucial forms:

1.  **Offline Federated Evaluation:** Devices compute metrics locally (e.g., Recall@K, nDCG@K for ranking quality) and then share only the aggregated results, protected with differential privacy noise.
2.  **Online A/B Testing:** Subsets of devices adopt new models to measure real-world impact on key performance indicators (KPIs) like CTR, engagement time, or conversion rates.

In addition to quality metrics (Recall@K, nDCG@K), we must continuously track **online metrics** such as latency, abandonment rate, and overall click satisfaction. Furthermore, **fairness and bias metrics** must be monitored to ensure consistent search quality across different languages and locales.

---

## **Stage 5: Productionization, Trade-offs, and Deployment**

When moving to the real world, we deploy a **hybrid system** that intelligently combines centralized and decentralized components.

### Deployment Flow

Global models, such as the base reranker weights and item tower embeddings, are distributed efficiently to devices using a **model registry**. On the user device, these models rely on efficient inference engines like CoreML or TensorFlow Lite, often using quantized formats (like int8) for minimal latency.

When a user searches:

1.  The device calculates the query embedding.
2.  It either performs local retrieval (if a partial index is cached on-device) or sends the embedding to the server to query the **global ANN index**.
3.  The top candidates are returned to the device, **reranked locally** using private personalization features, and then rendered.

### Training and Security in Production

Federated training happens periodically—perhaps **daily or weekly**—depending on how fresh the model needs to be. Crucially, only the gradient updates are ever transmitted, never the raw user data. These updates undergo clipping, noise injection, and encryption before being aggregated in a **Trusted Execution Environment (TEE)** prior to updating the global model.

Continuous production monitoring tracks model drift, ensures that latency budgets are met, and meticulously tracks the consumption of the **privacy budget ($\epsilon$)**.

### The Essential Trade-Offs

The design requires balancing conflicting goals:

*   **Personalization vs. Privacy:** Greater personalization requires utilizing more private user features, directly opposing strict privacy goals.
*   **Accuracy vs. Privacy Noise:** Increasing Differential Privacy (DP) noise strengthens privacy but often results in reduced model accuracy.
*   **Relevance vs. Latency:** Increasing the reranking model's complexity boosts relevance but risks exceeding the strict **on-device latency budgets**.

The most practical solution is a carefully balanced architecture. This often means freezing the heavy, server-side models and concentrating the personalization effort by only updating and fine-tuning small **adapter modules locally** on the device.

**In conclusion, the system is a hybrid federated architecture** where the global item encoder and retrieval index remain centralized, while the query encoder and ranking components are learned federatedly with differential privacy. This approach maximizes both personalization and privacy simultaneously.

---

## **Stage 6: Critical Design Questions for Next Steps**

Before finalizing the deployment plan, two clarifying questions are essential to ensure the system is aligned with real-world constraints.

1.  **What Privacy Commitments Must We Meet?** Asking *what differential privacy budgets ($\epsilon$, $\delta$) the organization currently adheres to* is critical. This number helps directly calibrate the fundamental trade-off between the level of noise applied and the resulting model performance.
2.  **What are the Speed Requirements?** Asking *what the exact latency service-level objectives (SLOs) are for the user experience*. The overall architectural design changes significantly depending on whether the system must enforce retrieval entirely on-device or if it can safely rely on a low-latency network call to the server’s ANN index.

These questions demonstrate a keen understanding of both the infrastructure limitations and the overarching product goals, ensuring the final search experience is both delightful and compliant with the company's privacy commitments.

***

> **The Privacy-Preserving Search Engine is like a master chef (the server) learning new techniques from hundreds of private home cooks (the devices). The chef never sees the specific recipes (raw data) used in any one kitchen, only the aggregated, noise-protected techniques (model updates) shared securely, resulting in a globally improved dish (search result) that still feels perfectly tailored to your individual taste (personalization).**