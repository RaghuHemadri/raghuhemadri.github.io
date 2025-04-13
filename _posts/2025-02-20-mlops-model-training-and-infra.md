---
layout: post
title: "MLOps: Model Training Infrastructure and Platform"
date: 2025-02-20 09:00:00
description: Infrastructure and platform requirements to support large model training.
tags: mlops
categories: ml
thumbnail: assets/img/train-infra.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

With the rapid evolution of machine learning, training models at scale has become an everyday challenge—from tweaking a small experiment to running full-blown multi-billion parameter training jobs. In our previous discussion, we explored techniques for training very large models. In this post, we will take a deep dive into the infrastructure and platform requirements that enable such tasks. By focusing on experiment tracking, versioning, reproducibility, and efficient scheduling, we aim to offer a guide for beginners while also providing detailed insight for the advanced practitioner.

## 1. Background and Motivation

### Distributed Training: The Foundation for Large Models
Modern machine learning pipelines increasingly rely on distributed training. For instance, training a model like OPT-175B requires splitting tasks over many nodes and managing different system components simultaneously. Distributed training is no longer just a technical challenge; it is an opportunity to design smarter infrastructure that scales with business needs.

### Why Infrastructure Matters
Behind every successful training operation lies robust infrastructure and an agile platform that supports a multitude of experiments. The infrastructure must handle everything from saving vital training logs to allocating scarce resources based on business priorities. Whether you’re dealing with diverse ML frameworks or managing hardware fluctuations, your platform must be prepared for all scenarios.


## 2. Core Components of Model Training Infrastructure

### Experiment Tracking and Monitoring
At the heart of any effective training platform is an experiment tracking system. Let’s break down its key capabilities:

- **Run Details and Reproducibility:**  
  An essential requirement is to save detailed logs for every training run. This includes:
  - **Model checkpoints:** Periodic snapshots to safeguard progress.
  - **Hyperparameters:** Every configuration used during training.
  - **Code version:** The exact codebase in use during the run.  
  These details ensure that experiments are reproducible and that teams can retrace their steps when fine-tuning or diagnosing issues citeturn0file0.

- **Monitoring ML Measures:**  
  Tracking important metrics such as loss, accuracy, and other custom measurements helps teams gauge training progress and diagnose any potential divergence early on.

- **System Health Observability:**  
  Monitoring isn’t limited to model performance. It extends to:
  - **Hardware health:** CPU/GPU utilization, memory usage, and thermal metrics.
  - **Software health:** Error alerts and performance logs of the underlying training software.
  - **Utilization metrics:** A real-time view of resource consumption that can highlight bottlenecks or underutilization.

### Infrastructure Health and Robustness
Beyond experiment tracking, the infrastructure must support the overall ecosystem:
- **Hardware Monitoring:**  
  Keeping an eye on hardware ensures that any potential failures or slowdowns are caught in time. This includes continuous health checks of GPUs, CPUs, and supporting systems.
- **Software Systems Monitoring:**  
  Alerting mechanisms and logging of software health prevent unforeseen issues that might derail a long training run.
- **Integration:**  
  A unified platform that cross-references experiment data with system logs empowers teams to make data-driven decisions on resource adjustments and job scheduling.

## 3. Job Scheduling for ML Training

Efficient scheduling is paramount when training multiple models concurrently or managing large-scale jobs that demand substantial computational power.

### What Makes a Good ML Training Scheduler?
A robust scheduler should ideally:
- **Effectively Allocate Resources:**  
  It must align resource usage with business priorities while ensuring that hardware isn’t sitting idle. For example, leveraging a job scheduler allows you to partition the available resources in line with specific workload demands.
- **Provide Usability and Flexibility:**  
  Users and administrators should have an intuitive interface that offers clear observability into job statuses and wait times.
- **Ensure System Fairness:**  
  Policies such as fairshare and priority scheduling help ensure that both large and small jobs are run according to their urgency and resource needs.

### The Basic Infrastructure of a Scheduler
At its simplest, a typical scheduling system comprises:
- **Head Node:**  
  The central orchestrator that manages the job queue and dispatches tasks.
- **Job Queue:**  
  The ordered list of jobs waiting to be processed.
- **Worker Nodes:**  
  The computational units that execute tasks.  
- **Storage:**  
  Persistent storage for saving model checkpoints and experiment logs.

Effective scheduling must also be designed to **avoid deadlock**—a scenario where two jobs are partially scheduled (for example, job A and job B both require two workers) but neither can progress because the remaining available resources are insufficient for either job.


## 4. Scheduling Policies: From FIFO to Gang Scheduling

Model training schedulers in modern infrastructures do more than simply run jobs in the order they are submitted. Let’s examine several core scheduling policies that can be fine-tuned to meet diverse training needs:

### FIFO (First-In-First-Out)
- **Overview:**  
  The simplest scheduling mechanism, where jobs are processed in the exact order they arrive.
- **Advantages:**  
  Predictable, easy to understand, and implement.
- **Limitations:**  
  May not be optimal if earlier jobs require significantly more resources or run much longer than subsequent jobs.

### Backfill Scheduling
- **Concept:**  
  Small or short-duration jobs are scheduled to fill in gaps between longer jobs.
- **Impact:**  
  This maximizes resource utilization and reduces wait times, provided the scheduler can estimate job runtimes accurately.

### Fairshare Scheduling
- **Philosophy:**  
  Ensures a “fair” allocation of resources by occasionally moving jobs ahead of earlier submissions if their users have historically consumed fewer resources.
- **Outcome:**  
  Promotes overall equity, especially in multi-team environments.

### Priority Scheduling
- **Mechanism:**  
  Jobs can be prioritized based on business needs. Higher-priority tasks may jump ahead of others, even if they arrive later.
- **Use Case:**  
  Critical business processes or time-sensitive experiments may preempt routine jobs.

### Preemption
- **Function:**  
  Allows the scheduler to pause lower-priority jobs to free up resources for those with more urgent needs.
- **Key Benefit:**  
  Ensures that critical tasks are not delayed, with the ability to resume paused jobs later.

### Time Slicing
- **Idea:**  
  A single computational resource can be time-shared by multiple jobs. Each job gets a slice of the total execution time.
- **Consideration:**  
  This approach must be carefully managed to prevent delays, especially in cases where inter-thread communication is critical.

### Gang Scheduling
- **Definition:**  
  Groups of jobs that need to run simultaneously (often because their tasks are interdependent) are scheduled together.
- **Scenario:**  
  Particularly useful in distributed training where synchronization across nodes is mandatory.

## 5. Resource Placement and Allocation Strategies

Efficiently placing jobs on available hardware can have a significant impact on training speed and cost-effectiveness.

### Placement Strategies
- **Spread:**  
  Distributes jobs across different nodes to minimize the risk of resource contention.
- **Pack:**  
  Concentrates jobs on fewer nodes to make sure that unused resources on a node do not remain idle.
- **Balanced Allocation:**  
  A hybrid approach where jobs are “packed” until a node reaches around 50% utilization before moving on to the next one. This helps to avoid overloading a single node while still making efficient use of resources.
- **Placement Groups and Bundles:**  
  Jobs can be grouped together if they benefit from co-location (for data sharing or inter-process communication), or conversely, be separated to reduce interference.
- **Object Locality:**  
  Some tasks might benefit from running on hardware that is physically close to a particular storage or another processing unit, thereby reducing latency.

These strategies ensure that jobs are allocated with precision, reducing resource hogging and addressing the variable nature of job runtime and requirements—especially critical in high-demand environments.

## 6. Real-World Case Studies

Drawing from recent case studies, we see how infrastructure design has been optimized to address very different needs across various organizations.

### Case Study 1: OPT-175 Logbook

The OPT-175 Logbook exemplifies the necessity for a rigorous experiment tracking system when dealing with massive language models. This case study reinforces several key points:

- **Comprehensive Run Documentation:**  
  Every training run needs to be meticulously recorded. In the OPT-175 setup, details such as model checkpoints, hyperparameter configurations, and code versions are saved. This level of logging is crucial not only for reproducibility but also for diagnosing anomalies during training. For instance, if performance degrades or a failure occurs, having a complete log ensures that you can pinpoint whether the issue stemmed from a change in hyperparameters or an update in the training code. citeturn0file0

- **Real-Time and Archived Monitoring:**  
  The system distinguishes between live monitoring and archival logging. Live monitoring helps track ML measures such as loss and other key performance indicators during the training process. In contrast, archived logs allow teams to look back at historical data, which can be invaluable when iterating models or conducting post-hoc analysis to improve training strategies.

- **Infrastructure Health Checks:**  
  Beyond the model’s performance metrics, the logbook also records the health of both hardware and software systems. This includes tracking GPU utilization, memory pressure, and even temperature readings—factors that can affect training stability. Likewise, the software’s performance—through error alerts and system logs—is monitored to quickly identify and correct any software-induced bottlenecks.

Together, these elements of the OPT-175 Logbook demonstrate a holistic approach: capturing every possible detail that could influence training outcomes. This meticulous tracking not only facilitates debugging and reproducibility but also informs improvements in infrastructure design for future large-scale training endeavors.

### Case Study 2: Nuro

Nuro’s training environment is designed to accommodate a more dynamic and diverse set of requirements, which makes its case study particularly interesting:

- **Support for Multiple ML Frameworks:**  
  Unlike systems that are built around a single framework, Nuro's platform is versatile enough to handle different machine learning libraries and environments. This flexibility is key in environments where teams might be experimenting with various algorithms and methodologies simultaneously.

- **High Resource Utilization with Low Wait Times:**  
  A significant challenge in busy training environments is ensuring that jobs do not get bottlenecked in queues. Nuro’s infrastructure tackles this by adjusting the job queue based on business priorities. High-priority tasks—potentially critical experiments or time-sensitive deployments—can jump ahead in the queue, while the scheduler still maintains fairness by ensuring that smaller jobs aren’t left waiting excessively.

- **State Management for Preempted Jobs:**  
  Another innovative aspect is the capability to manage job states when preemption occurs. If a high-priority job needs immediate resources, a lower-priority job might be paused (or preempted) rather than terminated. This means that when the high-priority job has been accommodated, the preempted job can resume from its last saved state—ensuring that computational efforts and time are not wasted.

- **Enhanced Observability:**  
  With clear insights into job status, wait times, and resource allocation, teams can quickly identify if resources are misallocated or if certain jobs repeatedly face delays. This observability is critical for iterative improvements and ensuring that the system remains agile in response to changing demands.

Nuro’s approach exemplifies how a scheduler must not only be adaptable but also intelligent enough to manipulate resource allocation in real time according to both technical demands and business priorities. citeturn0file0



### Case Study 3: Alibaba

Alibaba’s case study takes a deep dive into managing resource variability and ensuring optimal hardware utilization across a highly dynamic workload:

- **Fine-Grained Resource Allocation:**  
  Alibaba’s platform is tuned for scenarios where resource demands can change significantly over time. For instance, a job might require only a fraction of a GPU’s capacity rather than a full unit. This fractional allocation ensures that hardware is used as efficiently as possible, minimizing idle resources and reducing wastage.

- **Preventing Resource Hogging:**  
  In large, multi-user environments, certain jobs could potentially consume disproportionate amounts of resources. Alibaba’s system includes safeguards to avoid “resource hogging” by implementing scheduling policies that enforce fairness and balance. This might involve limiting the maximum resources that a single job can claim or adjusting the scheduling to favor smaller, shorter jobs when necessary.

- **Dynamic Scheduling Based on Job Runtime and Hardware Needs:**  
  One of the key innovations highlighted in Alibaba’s approach is the importance of predicting job runtimes and resource requirements. By knowing these factors in advance, the scheduler can make more informed decisions—allocating scarce resources to jobs that need them the most, and balancing out longer-running tasks with smaller, more agile jobs. This dynamic approach helps maintain high overall system utilization while preventing bottlenecks.

- **Effective Scarce Resource Management:**  
  In environments where high-demand resources like GPUs are in short supply, the ability to intelligently manage and allocate these scarce resources becomes paramount. Alibaba’s strategy of fine-grained allocation paired with proactive resource management ensures that critical jobs are not starved of necessary hardware, while overall system performance remains high.

Alibaba’s case study provides a detailed blueprint for how to achieve a high level of efficiency in a resource-constrained environment. By combining techniques like fractional GPU allocation, dynamic workload assessment, and preemptive scheduling adjustments, the platform ensures that both short and long-term projects can coexist without one overwhelming the system. citeturn0file0



## Bringing It All Together

While each case study emphasizes different aspects of the challenges in model training infrastructure, the common thread is clear: a well-designed platform must handle extensive logging, real-time monitoring, and sophisticated resource management. The OPT-175 Logbook stresses the importance of detailed, reproducible experiment tracking; Nuro demonstrates how adaptability and multi-framework support can streamline operations; and Alibaba’s example highlights the impact of precise resource allocation to maximize efficiency and prevent bottlenecks.

By learning from these diverse approaches, engineers and data scientists can design systems that not only address the immediate challenges of training very large models but also adapt to future demands in scalability, reproducibility, and resource optimization. Together, these insights form the backbone of modern ML training infrastructure, driving the next generation of AI research and deployment.



## 7. Looking Ahead: The Future of ML Training Infrastructure

The landscape of ML training infrastructure is evolving rapidly. Looking to the future, several trends will likely shape the next generation of training platforms:

- **Dynamic Resource Allocation:**  
  Improved predictive models and real-time analytics will enable even more precise allocation of computational resources.
- **Increased Automation:**  
  Advanced orchestration systems will automatically adjust scheduling policies based on current load, resource health, and business priorities.
- **Integration Across Frameworks:**  
  A unified platform that seamlessly supports multiple ML frameworks can further ease the burden on teams, making it easier to move between experiments.
- **Enhanced Observability:**  
  With better real-time monitoring tools, both infrastructure and experiment tracking will become more integrated, allowing for faster recovery when issues arise.



## Conclusion

Building a robust model training infrastructure and platform is a multifaceted challenge—one that spans from the simple logging of hyperparameters to the elaborate orchestration of distributed computing resources. The blend of detailed experiment tracking, comprehensive monitoring of both ML and system metrics, and sophisticated scheduling and placement policies creates an ecosystem where large models can be trained reliably and efficiently.

Whether you’re getting started with small experiments or orchestrating complex training workflows across multiple teams and hardware clusters, these insights into scheduling methodologies (FIFO, Backfill, Fairshare, Priority, Preemption, Time Slicing, and Gang Scheduling) and resource allocation strategies (Spread, Pack, Balanced, Placement Groups, and Object Locality) provide a strong foundation for designing a platform capable of scaling with your ambitions.

In a world where innovation is rapid and computational needs are ever-expanding, investing in a flexible, resilient training platform is not just beneficial—it’s essential for staying competitive in the machine learning landscape.

Below is the continuation of our in‐depth blog series on model training infrastructure and platform, expanding on real-world design choices, workload characterization, and advanced scheduling techniques drawn from industry case studies. In this second part, we delve into Nuro’s practical approach for scaling ML training and discuss insights from a large‐scale workload analysis of heterogeneous GPU clusters from Alibaba PAI. This continuation covers how sophisticated scheduling policies, dynamic resource sharing, and predictive task duration modeling pave the way for efficient large‐scale ML training.



## Scaling ML Training at Nuro: A Real-World Implementation

As an AI-first company, Nuro faces the challenge of running thousands of distributed training jobs daily on a spectrum of accelerator resources. To support this scale, the Nuro ML Scheduler was designed with several strategic goals:

### Design Goals for the Nuro ML Scheduler

1. **Multi-Cluster & Accelerator Diversity:**  
   Nuro’s infrastructure operates clusters in different geographic locations, each provisioning a diverse selection of accelerators—including various NVIDIA GPUs such as V100s, A100s, H100s, and even TPUs. Because accelerator availability can vary by location, the scheduler automatically selects an optimal cluster based not only on hardware availability but also on data locality and cost-effectiveness citeturn1file0.

2. **Policy-Based Resource Allocation:**  
   Unlike random assignment, resource allocation at Nuro aligns with project priorities. The scheduler employs policies that reserve accelerator “budgets” for business-critical tasks and ensures that resource distribution matches the urgency of the training jobs. Users can even mark high-priority jobs that trigger preemption of lower-priority tasks, maintaining operational SLAs.

3. **Reduction of Resource Contention & Deadlocks:**  
   Distributed training requires that all parallel workers begin simultaneously. The scheduler addresses potential deadlock scenarios—where partially scheduled jobs could stall progress—by enforcing grouping policies (gang scheduling) and by intelligently “packing” jobs onto machines with the required contiguous resources. This ensures that once a job starts, it completes without further waiting on resource availability.

4. **High Observability and Transparency:**  
   A core tenant of the Nuro ML Scheduler is providing clear, real-time visibility into the job queue and resource usage. An interactive dashboard lets ML engineers and administrators view job statuses, predict wait times, and monitor utilization trends. This observability creates a feedback loop whereby engineers can refine their job submissions to match available resources.

5. **Decoupled Scheduling from Training Orchestration:**  
   To support the ever-growing variety of ML frameworks—such as TensorFlow, PyTorch, or even JAX—the scheduler is decoupled from training orchestration. This design decision avoids tight coupling with any single training framework, thereby reducing deployment complexity and allowing rapid adaptation as new frameworks emerge.

### Key Features of the Nuro Scheduler

- **Automated Cluster Selection & Accelerator Matching:**  
  The scheduler automatically decides the best cluster based on accelerator type, availability, and data locality, so users don’t need to manage these complexities manually.

- **Job Queuing with Prioritization and Quotas:**  
  By using an in-memory queue refreshed from the jobs database, the scheduler supports dynamic ordering. Jobs receive one of several priority levels, and quotas are enforced to prevent a single team from monopolizing resources.

- **Smart Preemption and Reserving-and-Packing Policy:**  
  Critical tasks are guaranteed accelerator access through a preemptive strategy. For example, high-GPU tasks (such as those training advanced NLP models) are scheduled on high-end GPUs (e.g., V100/V100M32) using a reserving-and-packing approach. This means the scheduler first attempts to reserve the most capable hardware before falling back on less advanced options.

- **Simulation-Driven Optimization:**  
  Comprehensive simulation studies based on historical job traces have confirmed that these techniques not only reduce queueing delays but also improve overall job throughput and developer productivity citeturn1file0.

In practice, Nuro’s advanced scheduler has led to better SLAs for job start times and more effective resource utilization, allowing ML engineers to focus on developing models rather than wrestling with infrastructure.



## Workload Analysis and Scheduling in Heterogeneous GPU Clusters

To complement Nuro’s real-world experience, we now turn to a detailed workload characterization study from Alibaba’s PAI platform. This research (from the NSDI preprint by Weng et al.) analyzed a two-month trace from a production GPU cluster comprising over 6,000 GPUs, providing a rich understanding of MLaaS workloads in large-scale heterogeneous environments citeturn1file1.

### Key Findings from the Alibaba PAI Workload Trace

1. **Heavy-Tailed Workload Distribution:**  
   The trace analysis revealed that a very small set of users dominates the workload—approximately 77% of task instances come from the top 5% of users. Moreover, around 85% of task instances require gang scheduling, with many tasks demanding hundreds or even over a thousand GPUs simultaneously. This heavy-tailed distribution underscores the need for a scheduler that can handle both numerous small tasks and a few very large, high-resource jobs.

2. **Fragmentation and GPU Sharing:**  
   Most task instances request less than a whole GPU. Without GPU sharing, this would lead to severe underutilization. To address this, the PAI scheduler employs GPU sharing, allowing multiple tasks to time-multiplex a GPU. Simulation experiments showed that in peak hours, GPU sharing could reduce the total number of GPUs needed by up to 73%.

3. **Queueing Delays and Task Duration Prediction:**  
   Short-running tasks, despite their brevity, often suffer from lengthy queueing delays due to head-of-line blocking. Approximately 9% of short task instances spent more than half their overall runtime waiting to be scheduled. By leveraging the recurrence of tasks—over 65% of tasks are repeated—the researchers developed predictors using features such as task meta-information (e.g., command-line parameters, data sources) that achieved less than 25% prediction error for 78% of instances. Simulations comparing FIFO with shortest-job-first (SJF) scheduling based on these predictions demonstrated reductions in average task completion time of 63–77%.

4. **Spatial Variance in Resource Allocation and Utilization:**  
   The workload exhibited a heavy-tailed demand for CPU, GPU, and memory, but actual usage was generally much lower than requested. For example, while median GPU requests might be as high as 0.5 GPUs per instance, the median actual GPU utilization was only around 0.042 GPUs. This discrepancy partly arises from overprovisioning—in particular, many tasks request more vCPUs than they actually need. Additionally, network and I/O usage in distributed training tasks were low compared to compute resource usage.

5. **Challenges with CPU Bottlenecks:**  
   Although GPUs are the primary workhorses for training tasks, CPUs often become a bottleneck. Tasks that rely heavily on pre-processing or simulation (e.g., click-through rate prediction models or reinforcement learning workloads) experience significant slowdowns when CPU contention is high. The trace analysis found that machines hosting delayed task instances had notably higher CPU usage, whereas GPU utilization showed little correlation with delays.

### Advanced Scheduling Policies Derived from the Analysis

Based on these observations, the study highlights several challenges and opportunities:

- **Reserving-and-Packing vs. Load Balancing:**  
  For high-GPU tasks (e.g., training advanced NLP and image classification models), a reserving-and-packing policy—reserving high-end GPUs first before considering load-balanced alternatives—significantly reduces tail queueing delays. Simulation results indicated that for business-critical tasks, this approach cut average queueing delays by as much as 68%.

- **Multi-Resource Scheduling:**  
  Effective scheduling must balance the allocation of CPUs, GPUs, memory, and network resources simultaneously. The clear disparity between requested and utilized resources, and the observed impact of CPU contention on task performance, argues for a more integrated, multi-resource scheduling approach.

- **Handling Workload Heterogeneity:**  
  The varied nature of the tasks—from those that are GPU-intensive and require fast interconnects to others that are more CPU-bound—necessitates a scheduler that can dynamically adapt its policies. For instance, tasks needing strict GPU locality might be reserved for machines equipped with NVLink-enabled GPUs, while more flexible, smaller tasks can be distributed across less advanced nodes.



## Challenges and Opportunities for Future Research

Both the practical insights from Nuro and the extensive analysis from Alibaba PAI reveal several open challenges for future scheduling systems:

- **Mismatch Between Provisioned Resources and Actual Usage:**  
  There is a clear gap between what tasks request and what they actually consume. Fine-tuning this mismatch through more intelligent resource estimation and dynamic adjustments remains an active area of research.

- **Load Imbalance Across Heterogeneous Machines:**  
  High-end machines may often remain underutilized because they are reserved for critical tasks, while low-end machines become overcrowded. Developing fair and adaptive load-balancing algorithms that account for these differences can improve overall throughput.

- **Multi-Resource Interference and Contention:**  
  As many tasks are affected by CPU interference—even when the primary computation is GPU-based—schedulers must consider interference across all shared resources (CPU caches, memory bandwidth, I/O). This opens the door to multi-resource-aware scheduling algorithms that jointly consider contention and utilization.

- **Predictability and Elasticity in Scheduling:**  
  While task duration prediction has shown promise for recurring jobs, further work is needed to extend these techniques to highly variable workloads and to support elastic scheduling models that allow dynamic scaling of resources during training.



## Conclusion

In this continuation, we have explored how leading-edge implementations like Nuro’s ML Scheduler and detailed analyses from large-scale production clusters are pushing the boundaries of ML training infrastructure. By deploying advanced scheduling policies, automating cluster selection, supporting GPU sharing, and leveraging task recurrence for better duration prediction, organizations can achieve dramatically improved resource efficiency and reduced job wait times.

The insights gained from both Nuro’s experience and Alibaba’s extensive workload trace underscore the need for highly adaptive, multi-resource scheduling systems that can manage the complex interplay between CPU, GPU, memory, and network resources in heterogeneous environments.

As ML workloads continue to evolve in scale and diversity, addressing these open challenges will be crucial for maintaining high levels of throughput, ensuring fairness across users, and ultimately accelerating the pace of AI innovation.

Stay tuned for more deep dives into the future of ML infrastructure, where continued research and evolving best practices will further shape the landscape of scalable, efficient AI training.

Below is a very long, very detailed blog post that serves as a continuation of our deep dive into model training infrastructure and platforms. In this post, we shift our focus to the hands-on, coding side of our infrastructure—demonstrating how to set up an experiment tracking system using MLFlow and how to schedule and run distributed training jobs using Ray. Although the original guide includes many steps for provisioning resources on a cloud testbed, here we abstract away the underlying resource reservation details. Instead, we focus on the core components and their configuration, so you can replicate this setup in your own environment.

# Experiment Tracking and Distributed Training with MLFlow and Ray

Modern machine learning requires more than just powerful algorithms—it demands an infrastructure that enables reproducible research, efficient resource scheduling, and real-time observability across teams. In this continuation of our series on model training infrastructure and platforms, we explore a practical coding guide that uses MLFlow for experiment tracking and Ray for distributed job scheduling and training. We walk through the detailed steps from setting up the underlying services with Docker Compose to incorporating advanced techniques such as autologging, hyperparameter tuning, and fault tolerance.

## Overview and Architecture

As machine learning models grow larger and the number of experiments proliferate, managing training jobs and tracking their performance becomes crucial. Our architecture now comprises two main pillars:

- **Experiment Tracking with MLFlow:**  
  MLFlow is a powerful experiment tracking system that logs parameters, metrics, artifacts (such as trained models and images), and system-level metrics (e.g., CPU and GPU utilization). It scales from a simple, personal setup to a robust multi-user tracking server backed by a database and an object store.

- **Distributed Training with Ray:**  
  Ray enables you to distribute and schedule training jobs across multiple GPUs, ensuring efficient resource utilization. With modules like Ray Train and Ray Tune, you can scale your training processes, recover from failures via checkpoints, and run systematic hyperparameter optimization.

The following diagram outlines the logical flow:  
- **User Code / Jupyter Notebooks** interact with the MLFlow tracking server via its API.  
- **MLFlow Tracking Server System** is composed of three key components—a PostgreSQL database for structured experiment metadata, an object store (MinIO) for storing artifacts, and the MLFlow server itself.  
- **Ray Cluster** orchestrates distributed training jobs through a head node and multiple worker nodes, while providing dashboards (via Prometheus and Grafana) for observability.



## Experiment Tracking with MLFlow

MLFlow is at the core of our experiment tracking infrastructure. It enables reproducibility and comparative analysis of your experiments through detailed logging and version control of model artifacts.

### MLFlow System Components

The MLFlow tracking server is composed of:
- **PostgreSQL Database:** Stores structured data related to runs (e.g., start and end times, hyperparameters, metrics).
- **MinIO Object Store:** Serves as an S3-compatible storage backend for model artifacts and other large files.
- **MLFlow Tracking Server:** A central service with which your training scripts interact via the MLFlow Python API.

The orchestration of these components is typically handled using Docker Compose. For example, a simplified YAML file would define:
- A MinIO container exposing ports 9000 (API) and 9001 (UI).
- A PostgreSQL container for the MLFlow backend.
- An MLFlow container that depends on both MinIO and PostgreSQL services, ensuring they are healthy before MLFlow starts.

### Bringing Up the MLFlow Tracking Server

By using Docker Compose (see our configuration file reference citeturn2file0), you can launch the entire MLFlow tracking system with a single command:
```bash
docker compose -f docker-compose-mlflow.yaml up -d
```
This command pulls the MinIO, PostgreSQL, and MLFlow images; creates persistent storage volumes for the database and object store; and starts the MLFlow server so that it is accessible on port 8000.

Once running, you can access:
- The **MinIO dashboard** at `http://<SERVER_IP>:9001` for checking the object store status.
- The **MLFlow web UI** at `http://<SERVER_IP>:8000` where you can see your experiments organized by “experiment” and “run.”

### Tracking a PyTorch Experiment with MLFlow

In our example, we simulate training a convolutional neural network for a fictional startup called GourmetGram—a platform for photo sharing of food. The training script (available in the [GourmetGram repository](https://github.com/teaching-on-testbeds/gourmetgram-train)) uses a MobileNetV2 backbone and a classification head, training on the Food-11 dataset.

#### Integrating MLFlow in Your Code

To start tracking an experiment, first import MLFlow packages:
```python
import mlflow
import mlflow.pytorch
```
Then, configure MLFlow either with:
```python
mlflow.set_tracking_uri("http://<SERVER_IP>:8000/")
```
or by setting the `MLFLOW_TRACKING_URI` environment variable before launching the script. You can also set the experiment name:
```python
mlflow.set_experiment("food11-classifier")
```
Before beginning training, start an MLFlow run:
```python
with mlflow.start_run():
    # Log system metrics automatically
    mlflow.log_params(config)  # Log hyperparameters stored in a dictionary
    # Training code ...
    mlflow.log_metrics({"epoch_time": epoch_time, "train_loss": train_loss}, step=epoch)
    mlflow.pytorch.log_model(model, "food11_model")
```
This code logs the hyperparameters, model performance metrics per epoch, and the final PyTorch model as an artifact. It also automatically tracks host system metrics (CPU, GPU utilization) if you pass `log_system_metrics=True` to `mlflow.start_run()`.

#### Enhancing with MLFlow Autologging

If you use frameworks like PyTorch Lightning, MLFlow provides autologging functionality:
```python
if trainer.global_rank == 0:
    mlflow.set_experiment("food11-classifier")
    mlflow.pytorch.autolog()
    mlflow.start_run(log_system_metrics=True)
```
This ensures that only the primary process logs the relevant information, avoiding duplicate logging in distributed setups.

### Model Registry and MLFlow API

Beyond tracking experiments, MLFlow allows you to register models in a model registry, so you can version them and manage their lifecycle (staging, production, etc.). Once your training run completes, navigate to the “Artifacts” section of the run in the MLFlow UI, and click “Register model” to create a new model entry (e.g., “food11-staging”).

You can also interact with the MLFlow tracking server programmatically using the MLFlow Python API:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
experiment = client.get_experiment_by_name("food11-classifier")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.test_accuracy DESC"], max_results=2)
best_run = runs[0]
model_uri = f"runs:/{best_run.info.run_id}/model"
registered_model = mlflow.register_model(model_uri=model_uri, name="food11-staging")
```
This snippet demonstrates how to query runs, select the best-performing model based on a metric (e.g., test accuracy), and register it in the model registry for future deployment.



## Distributed Training and Job Scheduling with Ray

Now that we have an efficient experiment tracking system, let’s turn our attention to running distributed training jobs using Ray—a flexible framework that handles resource scheduling, distributed model training, and hyperparameter tuning.

### Understanding the Ray Ecosystem

Ray provides several key modules:
- **Ray Cluster:** Manages hardware resources across head and worker nodes.
- **Ray Train:** A simplified API for distributed model training.
- **Ray Tune:** Supports automated hyperparameter optimization by running multiple experiments in parallel.

With Ray, you can define a runtime environment using a JSON file (e.g., `runtime.json`) which specifies the Python dependencies via a `requirements.txt` file and sets necessary environment variables (like where to find your dataset).

### Deploying a Ray Cluster

Using Docker Compose, you can bring up a Ray cluster with a head node and multiple worker nodes. Depending on your GPU type (AMD or NVIDIA), you have separate Docker Compose configurations. For example, to launch a Ray cluster on NVIDIA GPU nodes, run:
```bash
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker compose -f docker-compose-ray-cuda.yaml up -d
```
This command starts containers for the ray-head and ray-worker nodes. You can verify the GPU availability on each worker by executing:
```bash
docker exec -it ray-worker-0 nvidia-smi --list-gpus
docker exec -it ray-worker-1 nvidia-smi --list-gpus
```
In our setup, we pass one GPU per worker to demonstrate efficient resource allocation and to enable fractional GPU resource allocation in later experiments.

A separate Jupyter notebook container (without GPUs) can be started to submit jobs to the Ray cluster:
```bash
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker run -d --rm -p 8888:8888 \
    -v ~/workspace_ray:/home/jovyan/work/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    --name jupyter \
    jupyter-ray
```
Access the Jupyter UI at `http://<SERVER_IP>:8888` to run job submissions and monitor experiments.

### Submitting Jobs to the Ray Cluster

Once your Ray cluster is live, you can submit a job (for example, a distributed PyTorch training script) using:
```bash
ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 1 --working-dir . -- python gourmetgram-train/train.py
```
The runtime environment file ensures that all worker nodes have the necessary Python packages installed and that environment variables (like `FOOD11_DATA_DIR`) are set.

If you mistakenly submit an infeasible job (e.g., a job requesting more GPUs than available), Ray will keep it in a pending state until resources are freed—providing clear feedback on resource demands via the Ray dashboard.

### Scaling, Fractional GPU Allocation, and Fault Tolerance

Ray Train makes it straightforward to scale your training jobs:
- **Scaling to Multiple Workers:**  
  Change your configuration in the training script from:
  ```python
  scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1})
  ```
  to something like:
  ```python
  scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1})
  ```
  This allows your job to run distributed training across two nodes.

- **Fractional GPU Use:**  
  If your training process does not fully exploit a whole GPU, you can specify fractional GPU resources in `scaling_config`, enabling you to pack more processes onto a node and increase cluster throughput.

- **Fault Tolerance with Checkpointing:**  
  By wrapping your training loop to periodically save checkpoints (using, for instance, `mlflow.pytorch.log_model` in PyTorch Lightning), Ray Train can automatically resume the training process on another worker if the current worker fails. In the training function, use:
  ```python
  checkpoint = ray.train.get_checkpoint()
  if checkpoint:
      # Load checkpoint and resume training
  else:
      # Start training from scratch
  ```
  This fault-tolerant mechanism is critical in production environments and ensures that long-running jobs can recover from hardware failures.

### Using Ray Tune for Hyperparameter Optimization

Ray Tune integrates seamlessly with Ray Train to support hyperparameter optimization. You simply modify your training script to wrap your training logic within a function that accepts configuration parameters. Then, define a configuration search space, for example:
```python
config = {
    "batch_size": tune.choice([32, 64]),
    "dropout_probability": tune.uniform(0.1, 0.8),
    # Include other hyperparameters
}
```
Submit this job using Tune’s API, which employs an early-stopping scheduler like ASHAScheduler to efficiently allocate resources:
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(max_t=config["total_epochs"], grace_period=1)
tuner = tune.Tuner(
    trainer,  # your Ray TorchTrainer instance or training function
    param_space={"train_loop_config": config},
    tune_config=tune.TuneConfig(metric="val_accuracy", mode="max", num_samples=16, scheduler=scheduler),
)
results = tuner.fit()
```
Ray Tune will automatically prune underperforming experiments and return the best hyperparameter configuration, all while allowing you to visualize progress in the Ray dashboard.



## Case Study: GourmetGram Food Classifier

Let’s tie all these components together with a real-world case study—the GourmetGram Food Classifier. In this example, a PyTorch-based model is trained to classify images in the Food-11 dataset into categories such as Bread, Dessert, and Meat.

### Experiment Workflow:
1. **Initial Non-MLFlow Run:**  
   Run the original training script to ensure that the code functions correctly in a distributed environment. This preliminary step helps verify hardware and software configurations.

2. **Integrate MLFlow Tracking:**  
   Switch to the MLFlow branch of your repository to incorporate logging functions. Compare experiment runs by logging parameters, metrics, and models.

3. **Upgrade to PyTorch Lightning:**  
   Transition the code to PyTorch Lightning for cleaner training loops and built-in support for MLFlow autologging. Ensure that only the primary process logs data by conditioning MLFlow calls on the `trainer.global_rank` variable.

4. **Deploy on Ray Cluster:**  
   Submit your training job to a Ray cluster to take advantage of distributed training across available GPUs. Experiment with scaling configurations and monitor the workload using the Ray dashboard.

5. **Optimize Hyperparameters with Ray Tune:**  
   Finally, use Ray Tune to run hyperparameter optimization experiments. Compare results side by side in the MLFlow UI to select the best model based on validation accuracy and other metrics.

This case study demonstrates the end-to-end process—from setting up the experiment tracking infrastructure with MLFlow to scaling and optimizing training using Ray Train and Ray Tune.



## Conclusion

The combination of MLFlow and Ray offers an integrated platform for managing large-scale model training experiments. With MLFlow, you can keep detailed records of each experiment run—logging everything from hyperparameters and performance metrics to system-level resource usage. Using Ray, you can distribute training tasks efficiently across GPU clusters, scale up or down as needed, implement fault tolerance with automated checkpointing, and leverage sophisticated hyperparameter optimization techniques.

This detailed guide has walked you through the practical setup and execution of these components, using a real-world case study—the GourmetGram Food Classifier. By following these steps, machine learning teams can not only accelerate model training but also gain valuable insights into their experiments, ultimately enabling faster iterations and more reproducible research.

Whether you are a beginner stepping into large-scale ML training or an advanced practitioner looking to optimize distributed workloads, the synergy of MLFlow and Ray provides a robust foundation for building your end-to-end model training infrastructure.

Stay tuned for future posts where we explore additional enhancements and the latest trends in model training infrastructure, further pushing the boundaries of scalable and efficient AI development.
