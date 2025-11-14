---
layout: post
title: "ML Sys Design: How Hybrid AI Delivers Speed and Privacy"
date: 2025-11-13 09:00:00
description: 
tags: system-design ml
categories: ml
thumbnail: assets/img/safari_ai.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# The Secret Brain Behind Your Smart Safari Assistant: How Hybrid AI Delivers Speed and Privacy

Have you ever wondered how your smart assistant can answer complex questions instantly while keeping your browsing private? The secret lies in a sophisticated **hybrid on-device and cloud-based LLM system** designed specifically for Safari. This system acts as a two-tier reasoning engine, balancing lightning-fast local processing with powerful, heavy-duty cloud synthesis.

***

## Stage 1: Problem Understanding (The Core Mission)

The primary goal is to build a Safari assistant that interprets user queries, decides whether the answer can be handled locally or requires fetching information from the web, and then generates responses that are both grounded in evidence and safe for privacy.

The core idea revolves around using a **lightweight on-device LLM** as the speed mechanism. This smaller model quickly handles query understanding and trigger decisions. If the task requires deep analysis, a more powerful **cloud or developer-kit LLM** steps in to perform heavier reasoning and synthesis.

### What the Assistant Handles:
The system processes natural language queries from various inputs—typed text, voice, or contextual cues within the browser. These queries might reference a specific webpage, selected text, a shopping site, or even past browsing context. The resulting output can be diverse, including a concise answer, a page summary, a suggested link, or a contextual action like “open this in Maps” or “compare prices”.

### The Strict Constraints Guiding the Design:
1.  **Speed (Latency):** Latency must be extremely low. On-device understanding and triggering must happen in **under 300 milliseconds** to ensure perceived responsiveness. Even when the cloud model is involved, the full interaction must complete within roughly two seconds.
2.  **Privacy (Central to Design):** User data, browsing history, and local content **must never leave the device in raw form**. Only minimal, aggregated, or redacted representations are permitted to be sent to the cloud.
3.  **Size (Hardware Fit):** The on-device model must be small—ideally **under three billion parameters** and quantized—to fit comfortably within the memory and power limits of Apple hardware.

In essence, this is a **two-tier reasoning system**: the small on-device model is the **planner and gatekeeper** that decides what to retrieve and whether to involve the cloud, while the remote LLM is the **synthesizer** that performs the more complex reasoning tasks once evidence is collected.

***

## Stage 2: High-Level Design (How the Brain Works)

The overall system is composed of five specialized subsystems working together: local data indexing, the on-device query understanding model, a retrieval layer, a cloud reasoning engine, and a monitoring/feedback loop.

### 1. On-Device LLM: The Instant Interpreter
When you ask a question, the **on-device LLM** immediately interprets it. It performs essential tasks like intent classification (is this a navigation query or a question?), domain routing, and deciding if retrieval is needed. For example, it decides if the query is navigational (“open NYT homepage”), informational (“who won the game”), or action-oriented (“summarize this page”). If external data is required, it creates a **retrieval specification**—a structured plan describing exactly what to fetch and from where.

### 2. Retrieval Layer: Fetching the Evidence
This layer executes the plan.
*   **Local Retrieval:** If the required content (like bookmarks or history) is on the device, it searches local embeddings stored in a lightweight Approximate Nearest Neighbor (ANN) index.
*   **Cloud Retrieval:** If external information is needed, only the redacted query specification is sent to a **cloud-side retriever**. This retriever uses sophisticated search across large indices covering different verticals (like news, shopping, or technical documentation).

### 3. Cloud/Dev-Kit LLM: The Deep Reasoner
Once relevant documents are gathered, the larger **cloud or dev-kit LLM** takes over. This powerful model reasons over the retrieved snippets and synthesizes grounded answers that include explicit citations. To maintain factuality and prevent hallucinations, the reasoning process is strictly constrained: the model is instructed to output only content that can be traced directly back to the evidence.

### 4. Response Orchestrator and Feedback Loop
The orchestrator merges results, enforces privacy and safety filters, and streams the final answer back to the user. Crucially, a telemetry and feedback loop collects interaction signals (such as clicks or refinements) in a privacy-preserving way. This data is later used via **federated learning** to improve how the system routes queries and classifies intent.

Conceptually, the data flow is robust: ingestion of local and remote data → storage in privacy-aware indices → on-device LLM understanding → retrieval (local or cloud) → cloud reasoning → answer synthesis → monitoring and retraining.

***

## Stage 3: Data Considerations (Fueling the Intelligence)

Training this hybrid system relies heavily on specific categories of labeled data:

*   **Intent Understanding:** Labels corresponding to query types, such as question answering, summarization, or navigation, sourced from annotated query logs.
*   **Retrieval Triggering:** Positive labels indicate cases where invoking web retrieval successfully improved satisfaction, while negative labels show queries that were better answered locally.
*   **Domain Routing:** Labels show which vertical (e.g., news or shopping) yielded the most relevant answers.

### Key Feature Processing:
The system relies on various features to make decisions. For example, the intent model uses textual embeddings from the query. The retrieval trigger model considers features like query length, the recency of similar searches, or whether matching content exists offline. Contextual factors like locale, device type, or connectivity state help determine whether to prioritize a local or cloud lookup. To maintain privacy, all textual features are embedded, and behavioral features are anonymized and normalized.

### Managing Scale and Privacy:
1.  **Handling Imbalance:** Operating across 150+ regions means data imbalance is inevitable—common intents like navigation will dominate. Techniques like focal loss or class-weighted training are vital to ensure minority categories (such as comparison or multi-hop reasoning) are not ignored.
2.  **Ensuring Generalization:** Dataset splits are time-based to prevent leakage. They are also stratified across locales and device types to ensure the models generalize well. Default fallbacks and feature imputation handle missing data (e.g., when a user has no browsing history).
3.  **Privacy By Construction:** Privacy transformations are mandatory at every stage. **No personally identifiable information (PII) can leave the device**. Queries are sanitized, identifiers are hashed, and telemetry is aggregated using protocols like differential privacy or federated learning.

***

## Stage 4: Modeling, Metrics, and Training (Building the Models)

The entire design is a complex combination of **multitask classification**, **retrieval ranking**, and **generative question answering**.

### The On-Device LLM (The Distilled Planner)
This model must simultaneously detect intent, decide on retrieval necessity, and potentially rewrite the query into a structured form. Because of strict size constraints, this is typically a **distilled and quantized version** of a much larger teacher model, possibly a 1–3 billion parameter transformer. It is trained via multitask instruction tuning. Efficiency is paramount: quantization (to int4 or int8) and execution on the Neural Engine ensure efficient inference while respecting the device’s thermal limits.

### The Cloud Model (The Reasoning Synthesizer)
The cloud model plays a complementary role as the core reasoning and synthesis engine. It can be significantly larger—tens of billions of parameters. It takes the retrieved evidence and the sanitized query to produce a grounded, citation-rich answer. To guarantee factuality, decoding can be constrained: **every claim must explicitly reference one of the retrieved passages**. A separate reranker model ensures the retrieved documents are optimally ordered before being passed to the generator.

### Training Strategy
Training utilizes a hierarchical approach. The powerful cloud model is trained using supervised fine-tuning on preference data and retrieval-augmented datasets. The small on-device planner is trained by distilling the decisions of the cloud model and using human-labeled supervision. **Federated learning** is key for personalizing intent prediction without ever sending raw examples off-device.

### Measuring Success
The system's performance is measured both offline and online:
*   **Offline Metrics:** Focus on technical correctness, including retrieval recall@K, intent F1 score, grounded answer accuracy, and latency/energy efficiency.
*   **Online Metrics:** Measure user interaction, such as task success rate, user dwell time, and the fraction of answers accepted without refinement.
*   **Safety Focus:** Continuous monitoring tracks the hallucination rate and groundedness.

### Common Challenges and Solutions:
*   **Hallucination:** Mitigated by strict grounding enforcement and source citation constraints.
*   **Data Drift:** Detected by monitoring the distributional shifts in intent embeddings, leading to periodic retraining.
*   **Cold Start (New Users):** Handled using heuristic routing rules until enough personalized data becomes available.

***

## Stage 5: Productionization, Trade-offs, and Deployment (Getting it to You)

For production deployment, the on-device model is optimized, packaged via Apple’s Core ML framework, and accelerated using the Neural Engine. It communicates with a local feature store that holds the user’s ANN indices. The large cloud model resides behind a retrieval and synthesis service that is designed to autoscale with traffic.

### Inference Pathways: The Hybrid Strategy
The hybrid strategy determines the inference path based on the trigger decision:
*   **Local Path:** For purely local queries, the response is generated entirely on-device, ensuring near-instant latency.
*   **Complex Path:** For difficult queries, the device sends a minimal retrieval specification to the cloud, receives the evidence-grounded result, and streams it back. This hybrid method is essential for balancing system capability and user privacy.

### The Trade-Offs (The Balancing Act):
1.  **Accuracy vs. Latency:** Invoking the cloud offers superior reasoning and accuracy but inherently increases the round-trip time.
2.  **Personalization vs. Privacy:** While richer behavioral features would improve the on-device model's relevance, they must be processed strictly locally to preserve privacy.
3.  **Recall vs. Safety:** Broader retrieval finds more results but increases content risks, requiring careful tuning with trust scoring and domain whitelisting.

### Ongoing Maintenance
Monitoring infrastructure tracks both technical performance (energy, latency) and behavioral metrics (groundedness, satisfaction). Retraining happens on a tiered schedule: on-device intent models update weekly via federated aggregation; retrieval models refresh every two weeks; and cloud reasoning models are fine-tuned monthly using fresh preference data.

The overall design philosophy is clear: the **on-device planner serves as the privacy-preserving brainstem**—it manages understanding, triggering, and gating. The **cloud model functions as the higher cortex**, only engaged when the query complexity demands its deep reasoning capabilities.

***

## Stage 6: Key Takeaway Questions (Focusing on Future Design)

To conclude the design process and inform further development, there are two fundamental questions regarding operational constraints:

1.  **Privacy Strictness:** How absolute are the privacy guarantees on data leaving the device? For example, are differential privacy-sanitized tokens or hashed URLs acceptable, or must the policy be strictly zero data transfer? The answer here dictates the ultimate retrieval design.
2.  **Operational SLOs:** What are the target energy and latency Service Level Objectives (SLOs) across various device classes? Additionally, how does the development team currently use A/B experiments to successfully balance model improvements against these crucial constraints? Understanding this operational context is essential for building models that are not just highly performant but also deployable at the required scale.