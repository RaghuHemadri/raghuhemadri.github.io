---
layout: post
title: "MLOps: Introduction"
date: 2025-01-23 09:00:00
description: Introduction to MLOps and discussion on why it is important.
tags: mlops
categories: ml
thumbnail: assets/img/mlops_intro.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# **From Prototype to Production: Building Reliable Machine Learning Systems**

## **Introduction**

The transition from a prototype machine learning (ML) system to a full-scale production deployment is one of the most critical—yet challenging—phases in machine learning operations (MLOps). While prototypes focus on proof-of-concept development, production systems must handle real-world complexities, including scalability, data integrity, fairness, and business objectives.

In this article, we'll explore the key differences between ML prototypes and production systems, examine scaling challenges, and share best practices for building robust machine learning pipelines.

---

## **Prototype vs. Production ML Systems**

Before deploying ML models to production, it's crucial to understand the fundamental differences between prototypes and production-ready systems.

| Aspect | Prototype ML System | Production ML System |
| --- | --- | --- |
| **Scale** | Small, experimental | Large-scale, serves many users |
| **Data** | Readily available toy datasets | Requires collection, cleaning, and continuous monitoring |
| **Privacy & Fairness** | Less emphasized | Critical for ethical AI |
| **System Design** | Typically a monolithic model | Often composed of multiple models interacting |
| **Objective** | Optimizing for ML metric (e.g., accuracy) | Needs to drive business KPIs and outperform current solutions |
| **Automation** | Minimal automation | Requires an automated pipeline for model iteration |

### **Key Takeaways**

- Moving from a prototype to production requires **robust data pipelines**, **continuous monitoring**, and **automation**.
- Business goals must drive the success of a production ML system, not just model accuracy.

---

## **Challenges in productionizing ML Models**

Scaling ML models beyond prototypes introduces several key challenges:

### **1. Data Management**

- **Data collection**: Unlike controlled experiments, real-world data is noisy and often incomplete.
- **Data validation**: Poor-quality data can degrade model performance.
- **Dynamic nature**: Unlike static datasets in prototypes, real-world data distributions shift over time, requiring continuous retraining.

### **2. Technical Debt in ML Systems**

A well-known 2015 NeurIPS paper, *Hidden Technical Debt in Machine Learning Systems*, highlights the extensive technical debt accumulated in ML systems. The model code itself represents only a small portion of the system, with most resources devoted to **data collection, verification, feature extraction, configuration management, monitoring, and infrastructure**.

### **3. Model Deployment & Monitoring**

- **Scalability concerns**: Serving models efficiently at scale requires robust infrastructure for model inference.
- **Model degradation**: Continuous performance monitoring is needed to detect drift.
- **Explainability & fairness**: Regulatory and ethical concerns require that models remain interpretable and unbiased.

---

## **Building Reliable Machine Learning Pipelines**

A well-structured machine learning pipeline is essential for productionizing ML models. Key properties include:

### **1. Automation**

A basic (Level 0) ML workflow involves:

1. Manual data collection and preparation
2. Model training and selection
3. Model evaluation and deployment

However, manual interventions are costly. A mature ML pipeline should incorporate:

- **Automated data validation**
- **Continuous model training and retraining**
- **Automated monitoring and deployment**

### **2. Key Qualities of an ML Pipeline**

Regardless of the application, a high-quality ML pipeline should ensure:

- **Velocity**: Fast iteration and experimentation on data and models.
- **Validation**: Early detection of issues before they become expensive to fix.
- **Versioning**: Ability to track and revert changes in data, model, and configurations.

### **3. Application-Specific Considerations**

The design of an ML pipeline depends on several key factors:

- **Data availability**: How much and what kind of data is available?
- **User needs**: Who will use the system, and how?
- **Scale requirements**: How many concurrent users/predictions need to be served?
- **Privacy & fairness**: How critical are interpretability and bias mitigation?
- **Compute resources**: What hardware infrastructure is available?

### **4. Case Study: Movie Recommendation Systems**

Consider two movie recommendation engines:

- A **physical mail recommendation system** updates recommendations **monthly** based on shopping history.
- A **streaming platform's recommendation system** updates recommendations **in real time** based on watch history.

Each requires different ML system designs:

| Feature | Physical Mail | Streaming Service |
| --- | --- | --- |
| **Update Frequency** | Monthly batch updates | Real-time personalization |
| **Data Type** | Shopping history | Watch and browse history |
| **Evaluation Metric** | Purchases over time | Click-through rate (CTR) |
| **Privacy Consideration** | Recommendations visible to all | Personalized for the user |

---

## **Conclusion**

Moving from a prototype ML system to a full production deployment requires thoughtful design, automation, and monitoring. A well-engineered ML pipeline:

- Ensures **scalability, reliability, and maintainability**.
- Reduces **technical debt**.
- Aligns with **business objectives** beyond just optimizing ML metrics.

As ML systems continue to evolve, adopting **best practices in MLOps** is essential for building **robust, efficient, and ethical AI solutions**.

---

# **From Zero Insight to Predicting Service Time: An MLOps Case Study**

## **Introduction**

In modern logistics and e-commerce, accurately predicting **service time**—the time spent delivering an order to a customer—is crucial for operational efficiency. Poor estimates lead to delivery delays, increased costs, and driver dissatisfaction.

Oda, an online grocery delivery service, initially relied on **static business rules** for service time estimates. This inflexible method lacked precision, prompting Oda to adopt a **machine learning (ML)-based approach** for dynamic predictions.

This blog details our **end-to-end MLOps journey**, from **data collection and model building to deployment and real-world testing**. We explore key challenges, ML's impact, and the vital role of **MLOps best practices** in maintaining robust predictive models.

---

## **1. Why Predicting Service Time Matters**

### **What is Service Time?**

Service time refers to **the time a delivery driver spends at a customer's location** before moving to the next stop. It includes:

- **Finding a parking spot**
- **Restacking the delivery vehicle**
- **Scanning the order**
- **Carrying groceries to the customer's door**
- **Potential interactions with the customer**

### **Why Service Time Prediction is Critical**

Service time comprises **50% of a driver's total workday**, making it a **key component of delivery efficiency**. Inaccurate estimates lead to:

- **Delayed deliveries**, negatively impacting customer satisfaction.
- **Driver stress**, leading to higher turnover rates.
- **Inefficient route planning**, increasing fuel consumption and costs.

Initially, Oda used a **fixed 7-minute service time per delivery** with **manual adjustments** based on experience. This rigid approach proved inaccurate and led to frequent miscalculations.

---

## **2. Data Collection: The Role of Geofencing**

### **The Challenge: Lack of Reliable Data**

The primary hurdle in building an ML model was **data unavailability**. Without **accurate historical service time data**, creating a robust predictive model was impossible.

### **Exploring Data Collection Methods**

We explored several options to measure actual service time:

- **Continuous GPS tracking***Pros*: Highly accurate*Cons*: Privacy concerns, high data storage cost
- **Manual timestamp logging***Pros*: Accurate, driver-controlled*Cons*: Requires additional driver effort
- **Google Maps timestamps***Pros*: Easily accessible*Cons*: Inaccurate due to routing discrepancies
- **Geofencing***Pros*: Automatic, minimal driver interference, privacy-friendly*Cons*: Requires precise location data, edge cases need handling

### **Choosing Geofencing for Data Collection**

We implemented **geofencing**, creating **virtual boundaries around customer locations**. The system **logs timestamps when drivers enter and exit these zones**, providing **automated service time measurements**.

### **Implementation Steps**

1. **Pilot Testing**
    - Installed geofencing on driver devices.
    - Compared geofence timestamps with manually recorded service times.
    - Ensured privacy compliance with data retention policies.
2. **Handling Edge Cases**
    - Preventing false triggers when drivers pass geofences without stopping.
    - Managing overlapping geofences for nearby customers.
    - Filtering out erroneous timestamps from GPS fluctuations.

Geofencing **eliminated manual logging**, **improved accuracy**, and **enabled large-scale data collection**.

---

## **3. Building the Machine Learning Model**

After gathering **two years of historical service time data** through geofencing, we transitioned from a **rule-based system to an ML model**.

### **Feature Engineering: Identifying Key Predictors**

We identified these crucial features affecting service time:

- **Order size** (weight, number of items, number of boxes)
- **Location factors** (urban vs. rural, parking difficulty)
- **Customer history** (past service times, building floor, elevator presence)
- **Time-based variables** (day of week, time of day)

### **Choosing the ML Model**

After testing various models, we selected **LightGBM**, a gradient boosting model, for its:

- **Speed**: Faster training and inference than deep learning models.
- **Interpretability**: Clear feature importance analysis for business insights.
- **Robustness**: Excellent handling of **nonlinear relationships**.

### **Hyperparameter Tuning**

We optimized model performance using **Bayesian optimization with Optuna** to tune:

- Learning rate selection
- Decision tree depth
- Feature weight adjustments

---

## **4. Model Evaluation and Real-World Testing**

After training and validation, we **deployed the model** in Norway's **Sandvika region** as our test area.

### **Key Metrics for Model Evaluation**

We compared our ML model against:

1. **Business Logic Model** (previous rule-based approach)
2. **Naïve Model** (using historical average service time)

Using **Mean Absolute Error (MAE)** as our primary metric.

### **Results: ML Model vs. Business Logic**

📌 **ML Model Outperformed the Business Logic Model**

- **Reduced MAE by ~30 seconds**.
- **Predictions showed greater stability and adaptability** across conditions.

📌 **Challenges & Observations**

- **Minimal impact on overall route precision**
    - Individual time estimates improved, but total route completion time stayed similar.
    - Individual errors often **balanced out across multiple stops**.
- **Performance fluctuations during peak periods**
    - Holiday seasons, especially Christmas, showed decreased accuracy due to **unusual shopping patterns**.

Despite these challenges, our model **clearly outperformed the rule-based approach**.

---

## **5. MLOps Challenges and Future Improvements**

### **MLOps Challenges Encountered**

1. **Data Drift**: Customer behavior changes and external factors (weather, holidays) affected predictions.
2. **Scalability**: New delivery regions required continuous model fine-tuning.
3. **Model Monitoring**: Performance degradation needed constant tracking.

### **Planned Future Enhancements**

🔹 **Improving Data Quality**

- Implementing **real-time anomaly detection** to catch **erroneous service time records**.

🔹 **Enhancing Model Adaptability**

- Introducing **adaptive learning** for seasonal variations.
- Incorporating **real-time weather data** to account for weather-related delays.

🔹 **Exploration vs. Exploitation Trade-off**

- Adding an **exploratory component** for **dynamic adaptation to new scenarios**.

---

## **6. Key Takeaways: The Role of MLOps**

### **Why MLOps is Critical in Real-World ML Applications**

1. **Automated Data Pipelines**: Reliable geofencing ensures consistent, accurate data collection.
2. **Scalable Model Deployment**: Efficient updates across new delivery zones.
3. **Continuous Monitoring**: Regular tracking of model drift and retraining needs.
4. **Cross-functional Collaboration**: Uniting **data scientists, engineers, and business teams** for ongoing improvement.

Despite **real-world ML deployment challenges**, our approach **showcases MLOps' power in optimizing operations**.

---

## **Conclusion**

Our shift from **rule-based to machine learning-based service time estimation** has **improved prediction accuracy** and **opened doors for future optimization**.

Key **MLOps lessons learned**:

✔️ **Data quality is the foundation of ML success.**

✔️ **Real-world ML deployment requires constant monitoring.**

✔️ **MLOps bridges the gap between research and production.**

This case study shows how **MLOps best practices drive logistics efficiency**. 🚀

---