---
layout: post
title: "ML Sys Design: ML Design Interview"
date: 2025-11-11 09:00:00
description: 
tags: system-design ml
categories: ml
thumbnail: assets/img/ml-sys-design.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# The Six Stages of the ML Design Interview

The following sequence dictates the interview flow, from initial concept clarification to production readiness, with suggested timing to ensure successful completion:

#### Stage 1: Fully Understanding the Problem (~5 Minutes)

This initial stage is foundational. If you are off by even half a degree on your course to the Moon, you are assured to never get there. Failing to fully understand the problem often leads to an overcomplicated solution or a failure to address the requirements completely. Therefore, dedicate about **5 minutes** to truly understand the question.

When faced with details you don't know (like exactly how many users Facebook has), the system recommends making educated assumptions to sound more competent rather than pestering the interviewer with continuous questions. For instance, you should state that "Facebook has about 3 billion users".

#### Stage 2: High-Level Design (The Rocket Blueprint) (~6 Minutes)

If you neglect the high-level design, you will spend significantly more time in back-and-forth explanations later. This stage is where many ML candidates fail because they immediately dive into deep, technical rabbit holes, showing an inability to handle the high-level design and wasting precious time, making recovery impossible.

Instead, you must create a **rocket blueprint**. You need to be very deliberate about building a **very high-level system diagram**. This diagram should include barely enough detail to address the main requirements of the problem. The technical focus here must be abstract; instead of specifying "XGBoost," you should think "model," leaving the specifics for a later stage. This approach is vital for avoiding most miscommunications between you and the interviewer. Candidates should also be familiar with the whiteboarding software used in the interview to avoid wasting precious minutes figuring out how to draw basic shapes.

#### Stage 3: Data Considerations (~8–9 Minutes)

This stage, where ignition happens and the rocket lifts off the launchpad, is a major component of your overall score. This section focuses on extensive data questions that need coverage in a short timeframe.

To succeed here, you must demonstrate an understanding of:
*   What your **labels** are and where they will be coming from.
*   What your **features** are and how to translate them into numbers.
*   How to **normalize** those numbers.
*   How to **split** your data set.
*   How you will address **data imbalance**, as data sets are "almost always" imbalanced.

Due to the limited time (8 to 9 minutes), you should discuss only a few representative features, rather than attempting to discuss every feature exhaustively.

#### Stage 4: Modeling, Metrics, and Training (~15–16 Minutes)

This stage covers the most important thing the interviewer is looking for and is considered critical. At this point, there is no longer time for clarification; you must make the most of your time.

Your technical task is to discuss **modeling metrics and training**. Additionally, you must address and propose solutions for common issues that may arise, such as **overfitting**, **cold start problems**, and **time travel problems**. This section is where you demonstrate machine learning knowledge specifically targeted to the scenario at hand. Although the material is broad, nearly all machine learning problems boil down to either **classification or regression**, and many problems can be solved with either one. Defining the problem as one of these two standard paths provides a safe and reliable path forward.

#### Stage 5: Productionization, Trade-offs, and Deployment (The One Small Step) (~5 Minutes)

This boost gets you all the way to the moon. Stage 5 is your opportunity to "nug the interviewer socks off" in about **5 minutes**. Interviewers sometimes focus on how to launch and operate a machine learning system in production. If you are an expert in specialized areas, such as **training parallelization** or **online evaluation techniques**, this is the time to showcase that expertise.

A key requirement in this stage is to demonstrate analytical thinking by identifying that there are often multiple ways to solve a problem. You must come up with **trade-offs** between at least two or more reasonable solutions and then make a **strong recommendation** for one of them.

#### Stage 6: Questions for the Interviewer (Remaining Time)

While the world focuses on the success of the mission (the first five stages), it is equally important to ensure the astronaut lands safely. This stage is for your benefit—to avoid breaking into a company that breaks your will to be in the industry.

Use the remaining time to ask the interviewer questions about working at the company. Since you only have time for **one or two questions**, ensure they are impactful.

---

### Typical ML Design Questions

Following these steps with the suggested timing is crucial for acing any ML design interview. The vast majority of questions for generalist positions in big tech come from only a couple of machine learning areas:

1.  **Harmful Content or Inappropriate Content Detection:** Examples include detective firearm in Amazon listing or nudity in a social media post. 

2.  **Design a Recommender System:** Examples include designing a Twitter timeline or an Amazon product recommendation system. 

These two question types are popular because they test a broad swath of machine learning knowledge and allow the interviewer to dive into many different technical aspects. For specialized roles, questions tend to be highly targeted (e.g., a vision specialist should expect to design a vision system).