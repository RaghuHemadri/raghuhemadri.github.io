---
layout: post
title: "ML Sys Design: YouTube Recommendation System"
date: 2025-11-11 09:00:00
description: 
tags: system-design ml
categories: ml
thumbnail: assets/img/youtube_reco.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# Building the YouTube Homepage Recommendation System: ML System Design

The design inquiry focused on constructing the YouTube homepage recommendation system, a common challenge in ML system design. As an everyday YouTube user, the immediate consideration is the **presentation of recommendations** and the **device type** used, as the homepage fundamentally shapes the user's journey through the website.

### Initial Scope and Clarifications

The first clarification sought was regarding device type. For the purpose of the design, the scope was limited to **desktop and mobile** platforms, excluding devices like the iPad.

**The Difference Between Desktop and Mobile**

A major difference between these two environments is the presentation density. On a desktop, a user may view 10 to 20 videos at once on the homepage. Conversely, a mobile page typically displays only one initial recommendation, requiring the user to scroll for further suggestions.

This distinction directly influences the choice of optimization metrics:

*   For the desktop use case, **MAP (Mean Average Precision)** is appropriate, as it measures the number of correct recommendations received within an assortment.
*   For the mobile use case, **MRR (Mean Reciprocal Rank)** is more suitable, as it is "top heavy," focusing on where the first correct recommendation appears in the ranked list.

These evaluation methods would vary the system design accordingly.

**Scale and Segmentation**

Next, clarification was sought on user scale: whether to design separate recommenders for highly engaged versus general users, or a single homogeneous system encompassing the scale of Google. The goal is to design an overall recommender system usable in any scenario where a user lands on the homepage, covering signed-in users, signed-out users, and users across many different continents.

**Geographical and Safety Constraints**

Since the system must cover users across different continents, geographical restrictions must be considered. It was determined that while the policies themselves are out of scope, the system architecture must incorporate additional logic in the **serving layer** to enforce these policies. This serving layer filter is essential for imposing policies before recommendations reach the user.

### System Architecture Diagram

The design proceeds by constructing the overall system architecture. The total recommender system is designed as a composite of offline (batch) and online (re-ranking) components.

**1. Offline Batch Processing**

The system begins with a **recommender system job** running in the backend. The output recommendations are fed into a **queue**, typically a **Kafka queue**, or written to static **batch files** stored in **GCS (Google Cloud Storage)**. Providing provisions for both ensures flexibility, as different model segments might run on varying refresh schedules (Batch 1, Batch 2, etc.).

A processing job listens to the queue, reads the recommendations, and stores them in a **cache**. These cached recommendations represent the batch part of the job. The cache is keyed by the user.

**2. Online Re-ranking and Contextualization**

The online part of the job applies on-the-floor computation, which typically consists of re-ranking.

**The Session Evolution Concept:**
When a user first lands on the homepage, they are served precomputed recommendations from the offline batch job. However, within the same session, if the user interacts (e.g., searches for five items, watches six videos, and returns to the homepage), their recommendations might change.

This post-interaction recommendation is handled by the online component. It takes the offline (static) recommendations and combines them with **contextual features** for the user.

*   The online component loads the **model weights** of a dedicated online model.
*   It connects to a **feature store** which contains the current context (e.g., search terms, recently viewed videos, viewing duration).

Together, the cached offline recommendations, contextual model weights, and features are processed by a logic component that produces the final re-ranked recommendation list for that user at that point in time. The initial landing page can be treated as a special case of this generic architecture, where the online model runs without any current contextual features.

Crucially, the **offline model** can afford to be complex and "deep" because it leverages all historical data, while the **online model** must be *lightweight* to minimize latency and ensure fast homepage loading. The recommendation system evolves within the same session due to this composite structure.

**3. The Serving Layer Filter**

Once the recommendations are produced, they pass through a **serving layer logic filter** before being presented to the user. This is where necessary policies and restrictions are imposed.

Examples of serving layer filters include:

*   Removing videos blocked due to copyright grounds.
*   Filtering videos unavailable in the user's geographical location.
*   Applying age appropriateness predictions, based on models that determine if content is suitable for a viewing session.

This serves up the final list of recommendations to the user.

### The Model and Data Pipeline (Offline Job)

The next step focuses on building the daily batch processing job which generates recommendations for potentially hundreds of millions of users.

**Modeling Choice: HRMN and Meta**

Traditionally, recommendations literature suggests extensive user segmentation (e.g., separate models for super-engaged, engaged, and non-engaged/incognito users). However, for this design, a modern approach is chosen: a single model that "sandwiches all this into one implementation".

The specific architecture chosen is the **HRMN (Hierarchical Recurrent Neural Network) and Meta** network, based on the *Temple: Contextual Recommendation Real Time* paper published at KDD 2020 by Ma, Narayanswami et al.. This model aims to cover all use cases under one roof.

System design steps are generally categorized into: **Data, Model, Evaluation, and Deployment**.

**Key Features Considered by the HRMN/Meta Model:**

The model guarantees coverage of several critical factors:

1.  **Sequential Model of User Interactions:** Modeling what a user will watch next based on their historical viewing sequence.
2.  **Contextual Information (Item Features):** Leveraging deep features of the video content itself, independent of the user. This includes the title, description, and content derived from deep learning models skimming the video content or processing **closed captions (CC files)**.
3.  **Hierarchical Session Features:** Recognizing that user activity over a day is not statistically independent. The model breaks down activity into hierarchical sessions, understanding that actions are influenced by previous chunks of action.
4.  **Cold Start:** Addressing the issue of recommending content to users with no prior interaction history.
5.  **Negative Sampling:** Defining what constitutes a negative interaction (e.g., distinguishing between a video not watched and a video actively disliked), often by defining **hard negatives** and **soft negatives**.

A general recommender system feature space is divided into **user features, item features, and user-item features**. While engaged users generate rich user-item features, non-interacting users rely heavily on item features. The HRMN/Meta model incorporates all three feature types to produce a single model suitable for all user types.

Ultimately, the HRMN model outputs the **probability of a user clicking on an item**.

### Recall Set Generation (Candidate Set Selection)

Since it is impossible to feed YouTube's entire catalog of billions of videos into the HRMN model for every user to get a click probability, an initial **Recall Set (Candidate Set)** must be generated.

A **multi-base recall logic** is used, inspired by concepts published by Walmart, to create a composite candidate set. This set might typically consist of around 300 videos, which are then passed to the HRMN for ranking.

**Example Recall Strategies:**

The composite recall set is built using multiple strategies:

1.  **History and Similarity:**
    *   Identify the last five channels the user watched.
    *   Expand this by adding the top five channels similar to those five (using a similar items model, perhaps from a sister team). This yields 25 channels.
    *   Add the **latest three published videos** from each of these 25 channels (acknowledging YouTube's time-sensitive nature), generating 75 videos.
2.  **Trending and Popularity:** Add the top 10 trending videos in the user's geographical area.
3.  **Genre Trending:** Add the top 10 trending videos within the user's favorite genre.
4.  **Live Content:** Include currently popular live events or live content the user typically watches.
5.  **Viral Videos:** Suggest super-popular content (e.g., sensational NBA playoff highlights) that might capture a user's interest regardless of their typical genre preference.

### Offline Inference and Refresh Rate

The architecture for the offline process involves:

1.  Data pipelines training the HRMN (generating model weights).
2.  A separate process generating the Recall Set using the recall logic.
3.  An **Offline Inference Script** running the HRMN weights against the Recall Set.
4.  A **Sort and Rank Script** producing the final recommendations list.

**Refresh Frequency:**
Due to the highly dynamic nature of YouTube, where the catalog expands rapidly, this batch job needs frequent refreshing. While in static systems, daily refreshes might suffice, YouTube recommendations require a much faster schedule, believed to be **every hour**.

The use of the **Kafka queue** is crucial for handling dynamic refresh rates. For instance, the model can be refreshed every hour during the day when traffic is high, but perhaps only once every six hours during the night when user activity and new data are low. The high rate of video addition on YouTube (far exceeding the removal rate) necessitates that the HRMN implementation is **parallel and frequently refreshable** to filter new content quickly.

For the **online re-ranking model**, a simple structure such as an **RNN (Recurrent Neural Network)** could suffice to re-rank the initial recommendations based on current session context and search terms.

### Evaluation and Deployment

Evaluation is divided into offline testing and online A/B testing.

**Offline Evaluation (Model Agnostic)**

An evaluation engine is required to measure standard, model-agnostic metrics.

**Standard Metrics Measured:** NDCG, MRR, MAP, Hit Rate, and AOC.

**Measurement Methodology (Temporal Split):**
To measure these metrics, the training data is temporarily split. The model is trained on a defined period (e.g., six months of data). It then predicts recommendations for a held-out period (e.g., one month of untouched data). The model's predictions (the **Prediction File**) are compared against what the user *actually* clicked or watched during that month (the **Goal Set**).

**Metric Details (Example: NDCG):**
For metric calculation, relevance must be defined. A simple definition is that the relevance of an item $R_i$ is 1 if the video is present in the Goal Set, and 0 otherwise.

**NDCG (Normalized Discounted Cumulative Gain)** focuses heavily on getting recommendations in the correct order. The formula is DCG (Discounted Cumulative Gain) normalized by IDCG (Ideal Discounted Cumulative Gain). This results in a score between 0 and 1, indicating recommendation quality.

**Metric Alignment with Device:**
As established during clarification, the optimization goal depends on the layout:
*   Optimizing for **MRR** yields a better model for **mobile**.
*   Optimizing for **NDCG** or **MAP** yields a better model for **desktop**.

These metrics are critical during hyperparameter tuning to compare model files for the two different layout requirements.

**Online Evaluation (A/B Testing)**

Once the model is offline evaluated and ready, it is launched via an A/B test. Given YouTube's scale, the rollout must be careful, following an **exponential ramp strategy**.

**Rollout Sequence:**
1.  **1% Traffic:** Initial "no harm test" to ensure key performance indicators (KPIs) do not "fall off a cliff".
2.  Slowly ramp up to **5%, 10%, 50%**, and eventually **100% traffic**.

**Key Performance Indicators (KPIs):**
KPIs are typically defined by product managers, but technical designers must consider two types of metrics during A/B testing:

1.  **Success Metrics (KPIs to Improve):**
    *   **Click-Through Rate (CTR)**.
    *   **Viewing Depth:** How many people who click the video view it fully or more than 50% (to avoid promoting "clickbaity" content).
    *   **Average Session Time (AST)**.
2.  **Fail-Safe Metrics (KPIs not to Reduce):** These are metrics that should never fall significantly. For example, overall revenue should not drop, even if session time is high, ensuring the platform is monetarily benefiting.

Once the A/B test demonstrates statistical significance on both the fence (fail-safe) metrics and the success metrics, the system proceeds to a 100% product launch.

---
*The design, encompassing the composite offline and lightweight online components, coupled with careful metric selection based on device type and a rigorous A/B testing rollout, forms a complete system designed for high scale and low latency in a highly dynamic environment like YouTube.*