---
layout: post
title: "ML Sys Design: Building the Ultimate Privacy-First, Multi-Modal Retrieval Engine"
date: 2025-11-13 09:00:00
description: 
tags: system-design ml
categories: ml
thumbnail: assets/img/ultimate_privacy_first_retrieval_engine.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# Spotlight Search Reinvented: Building the Ultimate Privacy-First, Multi-Modal Retrieval Engine

We are designing a powerful PyTorch prototype that aims to revolutionize device-wide search, making it instantaneous and intelligent. The goal is to index everything on a user's device—from text files to photos to code notebooks—and enable semantic search with minimal delay, all while balancing on-device processing against optional cloud assistance.

***

## Stage 1: What We're Building: Instant, Semantic Search

The fundamental objective is to allow users to type into Spotlight and immediately find any relevant item across their device using **semantic understanding**—moving past brittle keyword matches.

### Inputs and Outputs
*   **Inputs:** The system must handle **heterogeneous local artifacts**, including text files, PDFs, source code, code notebooks, images (with or without legible text), and short-form audio (voice notes).
*   **Query:** While initially focused on free-form text, the design is structured to accommodate future image or voice queries.
*   **Output:** Results must be instantly useful, presented as a ranked list. Each item needs a short preview and a **justification** explaining the match, such as a snippet, a bounding box over OCR text in a photo, or a transcript phrase for audio.

### Core Constraints
1.  **Speed is Essential (Low Latency):** Spotlight is an interactive experience, so we target a tight latency budget, aiming for a **sub-100 ms perceived delay** for the first results, ideally delivering a streaming sensation while the user is still typing.
2.  **Privacy-First:** We maintain a privacy-first posture. **Raw user content never leaves the device** unless the user explicitly opts into a cloud assist.
3.  **Scale:** We assume a typical personal corpus size of fifty to two hundred thousand items.
4.  **Efficiency:** The system is built for modern laptop or phone-class hardware and uses **opportunistic background indexing** when the device is idle or charging.

In essence, we are creating a multi-modal, privacy-preserving retrieval system that encodes local items into a **shared embedding space**, searches them using an approximate nearest neighbor index, and returns highly relevant, explainable results with extremely low latency.

***

## Stage 2: How It Works: The Multi-Modal Pipeline

The system initiates with a file-system watcher that monitors creations, modifications, and deletions. Each event is routed to a processing pipeline appropriate for its modality.

### Preprocessing Artifacts
*   **Text and Code:** These are normalized and lightly segmented. For code notebooks, we separate markdown from code cells and retain critical symbol information (like imports and function names) because these often guide developer searches.
*   **Images:** A compact OCR stage extracts text along with its coordinates. It also generates a general-purpose visual embedding.
*   **Audio:** Voice activity detection is used for energy trimming, followed by transcription into text with timestamps. We also produce a short audio embedding to capture non-linguistic properties.
*   **Local Store:** All these artifacts are saved into a small local store, which functions as both a metadata registry and a pointer map connecting to vectors and previews.

### The Unified Embedding Engine
Embedding is handled by a PyTorch service that loads three small encoders—one each for text/code, images, and audio. These encoders project the content into a **single, unified embedding space** via a light multi-layer projection head.
*   **On-Device Efficiency:** To ensure efficient execution on the device, these models are exported with TorchScript and **quantized to int8**.
*   **Indexing:** The resulting vectors are maintained in a FAISS HNSW index. This index exists both per-modality (useful for diagnostics) and as a unified index (for cross-modal retrieval).

### Query Execution and Fusion
When the user types a query, their text is encoded using the lightweight text encoder.
1.  **Retrieval:** An approximate nearest neighbor search returns a candidate set.
2.  **Fusion:** A fast fusion step is applied. This combines the raw embedding similarity score with pragmatic signals such, as filename matches, BM25 scores (for available text), and recency boosts.
3.  **Re-ranking (Optional):** If the latency budget allows, a tiny cross-encoder is used to re-rank the top slice, refining borderline cases. If time is too tight, the bi-encoder signal and simple fusion are sufficient.
4.  **Feedback Loop:** The serving layer returns results with justifications (e.g., highlighted OCR regions, transcript phrases) and records on-device feedback, including clicks, dwell time, and query reformulations.

In a strictly opt-in cloud-hybrid variant, heavier tasks (like OCR or ASR) or applying a larger re-ranker could be offloaded, but raw user content always remains on the device.

***

## Stage 3: The Fuel: How We Teach the System What's Relevant

### Data Collection and Labeling
In a live product environment, the most valuable labels come from actual **user behavior**.
*   **Positive Signals:** A strong positive signal is a click quickly followed by opening the file or lingering in the preview.
*   **Negative Signals:** Negative signals include rapid "pogo-sticking" (bouncing back immediately to the results list) or immediate query reformulation.
*   **Preference Data:** Pairs of queries where the first attempt failed but the second succeeded provide natural preference data for pairwise training.

When building a prototype without live user traffic, we must **bootstrap with pseudo-labels**. This means treating exact keyword matches and symbol matches as positive examples, adding time-decayed weights for recency, and generating "hard negatives" from nearest neighbors that users failed to select.

### Features: Content and Context
Core features start with embeddings that accurately reflect content:
*   **Text/Code:** Embeddings capture semantics at the sentence or chunk level. Code search is enhanced by incorporating symbol inventories and lightweight static cues, which helps users find items like "that file with the function I wrote last week".
*   **Images:** Representations include a global embedding for overall content, coupled with OCR-extracted tokens and coordinates for precise highlighting.
*   **Audio:** Provides an indexable transcript and an acoustic embedding useful for matching short voice notes even when the transcript is noisy.
*   **Metadata:** This plays a critical role, as context often explains user intent better than raw content. Key metadata includes path segments, the application that created the file, EXIF tags for photos, and creation/modification times.

### Document Handling and Rigor
*   **Long Documents:** Because documents can be lengthy, they are split into manageable token-length chunks with slight overlap. A mapping back to the parent file ensures we can deduplicate results and present the best possible slice.
*   **Normalization:** Unexciting but essential tasks include standardizing Unicode, identifying language to select tokenizers, and cautiously removing boilerplate (to prevent over-indexing headers and footers).
*   **Weak Signals:** We insist on minimum confidence thresholds for image OCR and ASR. If signals are weak, the system transparently falls back to using filename and sparse matching.
*   **Footprint:** Even with one hundred thousand items, a 768-dimensional vector per item can be **product-quantized to just a few megabytes**, leaving ample space for indexes and metadata on modern devices.

***

## Stage 4: The Engine: Retrieval and Refinement

At its core, this is a retrieval-and-ranking challenge across various modalities.

### Modeling Architecture
The most practical starting point is a **bi-encoder architecture**. This design uses separate encoders for text/code, images, and audio, aligning them through a shared projection into a unified space.
*   **Serving Advantage:** This allows us to precompute all item vectors and store them in an HNSW index, enabling queries to be answered extremely fast with a single embedding operation and an approximate nearest neighbor search.
*   **Limitation & Solution:** The bi-encoder cannot inspect fine-grained cross-interactions between a query and a candidate's full content. Therefore, a small latency budget is reserved for an **optional cross-encoder** to refine the top few dozen results when ambiguity is high.

### Training Process
Training is implemented in two critical layers:
1.  **Modality Alignment:** We use a contrastive objective to align modalities. Naturally corresponding pairs (e.g., a notebook cell and its saved chart screenshot) are pulled closer in the embedding space, while sampled negative examples are pushed away.
2.  **Fusion Learning:** We then learn a simple fusion mechanism that combines the embedding score with sparse and metadata signals, which provides measurable quality gains at minimal operational cost.

**Hard negative mining** from the live index is crucial, ensuring the model can distinguish between highly confusable neighbors that an interactive user would actually encounter. Scores are calibrated using temperature scaling for consistent threshold behavior as models evolve.

### Metrics and Evaluation
Evaluation must rigorously cover both quality and performance.
*   **Quality (Offline):** We track quality metrics like nDCG and MRR at various cutoffs, demanding that improvements be consistent across all modalities and languages.
*   **Performance (Operational):** Metrics critical to the user experience include median and tail latency for the full query path, the memory footprint of the index, and the energy consumed during background indexing.
*   **Usefulness (Online):** In controlled tests, we monitor click-through rates at top positions, the rate of query reformulation, and time-to-first-open, as these correlate strongly with perceived usefulness.

### Addressing Pitfalls
We must guard against classic modeling issues:
*   **Temporal Leakage:** We ensure that user clicks only influence models trained on older interaction windows.
*   **Cold-Start Items:** New items rely more heavily on content embeddings and metadata until user interactions accumulate.
*   **Language Shifts:** Multilingual and code-mixed text are routed through appropriate tokenization and normalization.
*   **Noisy Transcripts:** Since OCR and ASR can be noisy, we track their confidence and use transparent fallbacks to filename and BM25 to prevent over-trusting bad transcripts.

### The Speed Secret
The entire process is optimized for speed:
*   An int8-quantized text encoder can typically embed a query in a **handful of milliseconds** on CPU.
*   An HNSW search over one hundred thousand items for the top candidates is completed within **single-digit milliseconds** when tuned.
*   The fusion step is essentially free.
*   The cross-encoder is the main discretionary cost, but by making it **adaptive** (triggered only when the embedding score distribution suggests ambiguity), we preserve snappy interactions without sacrificing quality on difficult queries.

***

## Stage 5: Launching Spotlight: On-Device First

The cleanest path to deployment begins entirely **on device**.

### Production Deployment
We ship the TorchScripted, quantized encoders and maintain the FAISS index locally. Indexing and compaction are scheduled to run when the operating system confirms the device is **idle and preferably charging**.
*   **Safety and Monitoring:** A signed model registry allows for safe rollout and rollback. A small, on-device telemetry loop (constrained to privacy-preserving aggregates) monitors latency, crash rates, and quality proxies.
*   **Cloud-Hybrid Path:** An opt-in cloud-hybrid system can be introduced later for performing heavier tasks (like complex OCR/ASR or applying a more expressive re-ranker for tough queries). Any such assistance must be **explicit, explainable, and revocable**, with raw content retained on the device and only differentially private gradients or encrypted vectors leaving it.

### Key Design Trade-offs
1.  **Latency vs. Accuracy:** The bi-encoder-only path is lightning fast, but the cross-encoder boosts quality; thus, the cross-encoder must be budgeted and adaptive.
2.  **Privacy vs. Assistance:** Strict on-device processing guarantees absolute user privacy, but certain quality jumps (especially for complex text or low-resource languages) are easier to achieve with cloud assists.
3.  **Battery vs. Freshness:** Being eager with indexing improves result freshness but increases energy cost; being cautious saves battery but risks stale results.

### The Recommended Path
The recommendation is to first ship a **fully on-device MVP** that relies on bi-encoder retrieval, simple and robust fusion, and transparent explanations for matches. This ensures the system meets the tight latency target and earns user trust. In a second phase, we can introduce the adaptive cross-encoder, expand modality coverage, and explore the opt-in cloud-hybrid lane and federated fine-tuning (designed to improve the shared projection head without collecting raw content).

***

## Stage 6: Design Decisions We Need to Confirm

Two critical questions remain to finalize the system design:
1.  **Privacy Boundaries for Derived Data:** What are the hard privacy boundaries for derived artifacts like vectors or differentially private aggregates? Can these elements leave the device under explicit consent, or must *all* machine learning remain strictly local?.
2.  **On-Device Experimentation Infrastructure:** How mature is the existing experimentation framework on the device? Can we quickly toggle features (like adaptive re-ranking) and collect privacy-safe outcome metrics to rapidly iterate on ranking, or should we plan for a slower cadence using offline proxies while the necessary on-device A/B infrastructure is built?.
