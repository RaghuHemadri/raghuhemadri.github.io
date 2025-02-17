---
layout: post
title: "MLOps: DevOps for ML systems"
date: 2025-02-06 09:00:00
description: Overview of the blog series
tags: mlops
categories: ml
thumbnail: assets/img/devops.webp
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

In this post, we explore the foundational principles of **DevOps** and how they apply to software development and deployment. From there, we delve into **MLOps**, extending DevOps principles to machine learning systems. The lecture also include a deep dive into **cloud-native computing**, the **ML system lifecycle**, and the **organizational capabilities** required to support large-scale ML deployment, using **Uber's Michelangelo platform** as a case study.

## **Understanding DevOps in Software Development**

Modern software development has evolved through three primary methodologies:

- **Waterfall**: A rigid, sequential process where each stage must be completed before moving to the next. This method often results in inefficiencies due to its lack of iterative feedback loops.
- **Agile**: Introduces iterative development, emphasizing rapid feedback but still primarily focused on code completion rather than operational aspects.
- **DevOps**: Integrates operations into development, enabling a **continuous loop** through automation and collaboration. This methodology focuses on rapid, reliable software delivery by reducing silos between development and operations teams.

### **Key DevOps Principles and Technologies**
- **Continuous Integration/Continuous Delivery (CI/CD)**: Automating the process of integrating code changes and deploying them into production.
- **Version Control for Everything**: Storing all code, configuration, and infrastructure definitions in version-controlled repositories to ensure traceability and collaboration.
- **Infrastructure as Code (IaC)**: Managing infrastructure through declarative or imperative code instead of manual configurations, ensuring consistency and reproducibility.
- **Proactive Monitoring and Logging**: Implementing automated alerts and observability for applications and infrastructure to detect and mitigate issues before they impact users.

### **Git Workflow in DevOps**
A fundamental tool in DevOps is **Git**, which supports:
- **Staging changes** (`git add`)
- **Committing code** (`git commit`)
- **Pushing updates to remote repositories** (`git push`)
- **Pulling changes from the repository** (`git pull`)

These workflows facilitate collaboration and prevent issues such as configuration drift.

### **Measuring DevOps Success: DORA Metrics**
DevOps Research and Assessment (DORA) has identified four critical performance metrics:

1. **Deployment Frequency** – How often new code is released to users.
2. **Lead Time for Changes** – The time from code commit to deployment.
3. **Change Failure Rate** – The proportion of releases that introduce issues requiring fixes.
4. **Time to Restore Service** – The time needed to recover from failures and incidents.

High-performing DevOps teams optimize these metrics, ensuring rapid yet stable software releases.

## **Cloud-Native Computing: DevOps Meets Cloud**
Cloud-native computing extends DevOps principles by leveraging cloud-based infrastructure for scalability, flexibility, and automation. Key concepts include:

- **Immutable Infrastructure**: Rather than updating existing infrastructure manually, changes are made by provisioning new instances with desired configurations.
- **Microservices**: Applications are composed of small, independent services that interact via APIs, allowing for modular scaling and deployment.
- **Containers**: Containerization (e.g., using Docker and Kubernetes) ensures consistency between development, testing, and production environments.

## **How Machine Learning Systems Differ from Traditional Software**
Unlike conventional software, ML systems require additional components:
- **Code**: Traditional software development components.
- **Infrastructure**: Cloud and on-premises hardware to support ML workloads.
- **Models**: Trained machine learning models that must be versioned and updated.
- **Data**: Continuous inflow of new training data requiring preprocessing and monitoring.

While DevOps provides mature practices for managing code and infrastructure, it does not fully address the challenges of managing models and data, necessitating **MLOps**.

## **The MLOps Lifecycle**
A machine learning system follows a structured lifecycle:

1. **ML Development**: Includes data exploration, feature engineering, and model prototyping.
2. **Continuous Training**: Automates model training, validation, and registration in a model store.
3. **Training Operationalization**: Integrates trained models into production workflows through CI/CD.
4. **Model Deployment & Serving**: Uses techniques such as **canary releases, shadow testing, and A/B testing** to safely deploy models.
5. **Monitoring & Logging**: Tracks data drift, model decay, and performance degradation.

## **Case Study: Uber’s Michelangelo Platform**
Uber’s **Michelangelo** platform is a production-scale MLOps system that supports the end-to-end machine learning lifecycle. The platform was designed to enable Uber to efficiently build, deploy, and maintain thousands of ML models in production.

### **Key Components of Michelangelo**
- **Data Processing**: Supports structured and unstructured data, feature engineering, and batch/stream processing.
- **Experimentation & Training**: Provides model development workflows with automated tracking, hyperparameter tuning, and distributed training.
- **Model Deployment & Serving**: Offers scalable, low-latency model serving with support for **canary deployments**, **A/B testing**, and **shadow releases**.
- **Monitoring & Feedback Loops**: Continuously tracks performance, drift, and anomalies, triggering retraining as needed.

### **Detailed Workflow in Michelangelo**
1. **Data Ingestion**: Michelangelo pulls in massive amounts of data from Uber’s operational systems, preprocesses it, and stores it in a data lake.
2. **Feature Engineering**: Automates feature extraction and transformation to create high-quality ML datasets.
3. **Experimentation**: Data scientists can use Jupyter notebooks and version control to iterate on models.
4. **Model Training & Validation**: Supports distributed training using TensorFlow, PyTorch, and other ML frameworks.
5. **Model Deployment**: Provides multiple deployment strategies, including batch scoring, real-time inference, and multi-model ensembles.
6. **Monitoring & Model Drift Detection**: Continuously tracks model accuracy, detects performance degradation, and triggers retraining when needed.

### **Real-World Applications at Uber**
- **Dynamic Pricing**: Predicting demand fluctuations to optimize fare adjustments.
- **ETA Prediction**: Enhancing arrival time estimates using real-time and historical trip data.
- **Fraud Detection**: Identifying fraudulent activity using anomaly detection models.
- **Uber Eats Recommendations**: Providing personalized restaurant and dish suggestions based on user behavior.
- **Autonomous Vehicle Research**: Supporting AI models for self-driving car development.

By integrating DevOps practices with robust MLOps capabilities, Michelangelo ensures that Uber’s models remain scalable, reliable, and continuously improving.

## **Looking Ahead: Scaling MLOps**
For the remainder of the semester, we will explore each core capability in detail. Next week, we will focus on **large-scale training**, a crucial aspect of optimizing model performance.
