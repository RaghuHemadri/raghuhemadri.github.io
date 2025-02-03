---
layout: post
title: "MLOps: Cloud Computing Definitions"
date: 2025-01-30 10:00:00
description: "Detailed explanations of concepts from `MLOps: Cloud Computing` blog"
tags: mlops
categories: ml
thumbnail: assets/img/mlops-cloud-def.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# **Block, File, and Object Storage**

Storage solutions in computing can be categorized into **Block Storage, File Storage, and Object Storage** based on their data structure and access methods.

---

## **1. Block Storage**

### **Definition**

Block storage splits data into **fixed-size blocks**, each with a unique identifier. When needed, the system retrieves and assembles these blocks to reconstruct the complete file.

### **Key Characteristics**

✅ Low latency, high performance

✅ Used for structured data (databases, VMs, etc.)

✅ Requires an OS or filesystem to manage file structure

✅ Typically found in SAN (Storage Area Network) systems

### **Example**

- **Amazon EBS (Elastic Block Store)**
- **Google Persistent Disk**
- **iSCSI (Internet Small Computer Systems Interface)**

### **Use Case**

- **Databases** (MySQL, PostgreSQL, etc.)
- **Virtual machines** (VMware, KVM)
- **High-performance applications**

---

## **2. File Storage**

### **Definition**

File storage organizes data in a **hierarchical structure** of directories and subdirectories, managed by **a file system** such as NTFS, ext4, or NFS.

### **Key Characteristics**

✅ Easy to navigate (like a local disk structure)

✅ Supports file-level access

✅ Requires a filesystem to manage organization

### **Example**

- **Network File System (NFS)**
- **Server Message Block (SMB)**
- **Amazon Elastic File System (EFS)**

### **Use Case**

- **Shared drives** in organizations
- **File servers**
- **Media storage (videos, images, documents)**

---

## **3. Object Storage**

### **Definition**

Object storage stores data as **objects** in a **flat namespace**, where each object contains:

- The **data** itself
- **Metadata** (extra details like owner, timestamps, etc.)
- A **unique identifier (UUID)**

### **Key Characteristics**

✅ Scalable to petabytes and beyond

✅ Best for unstructured data (videos, backups, logs)

✅ No traditional file hierarchy (flat structure)

✅ Uses APIs for access (e.g., HTTP REST API)

### **Example**

- **Amazon S3 (Simple Storage Service)**
- **Google Cloud Storage**
- **Azure Blob Storage**

### **Use Case**

- **Cloud storage (Google Drive, Dropbox, etc.)**
- **Backup and disaster recovery**
- **Content delivery networks (CDNs)**

---

### **Comparison Table**

| Feature | Block Storage | File Storage | Object Storage |
| --- | --- | --- | --- |
| **Structure** | Divided into blocks | Hierarchical (folders) | Flat (key-value store) |
| **Performance** | High | Moderate | Moderate |
| **Access** | Low-level (disk I/O) | File-based (NFS, SMB) | API-based (REST, S3) |
| **Best For** | Databases, VMs | File sharing, home directories | Cloud storage, backups, multimedia |
| **Scalability** | Limited | Moderate | High (Petabyte scale) |

---

## **Which One Should You Choose?**

- **Use Block Storage** for high-performance applications like databases and virtual machines.
- **Use File Storage** when you need shared access to files, like a company network drive.
- **Use Object Storage** for large-scale, unstructured data like images, videos, and backups.

# **Virtual Networks, Load Balancers, and Security Groups Explained**

These three networking concepts form the foundation of **cloud computing and distributed systems**, providing essential **connectivity, scalability, and security**.

---

## **1. Virtual Networks (VNet/VPC)**

### **Definition**

A **Virtual Network (VNet)** or **Virtual Private Cloud (VPC)** is a **logically isolated network** in the cloud that enables secure communication between resources like VMs, databases, and storage.

### **Key Features**

✅ **Subnetting** – Dividing a network into smaller segments (subnets)

✅ **Private & Public IPs** – Enables internal and external communication

✅ **Network Peering** – Connects different virtual networks

✅ **Routing** – Defines how traffic flows between resources

### **Example**

- **AWS Virtual Private Cloud (VPC)**
- **Azure Virtual Network (VNet)**
- **Google Cloud Virtual Private Cloud (VPC)**

### **Use Case**

- Creating **isolated environments** for applications
- Connecting **on-premises networks** to the cloud via **VPN or Direct Connect**
- Implementing **microservices architecture** with different subnets

---

## **2. Load Balancers**

### **Definition**

A **Load Balancer** distributes incoming network traffic across multiple servers to ensure high availability and reliability.

### **Types of Load Balancers**

1. **Layer 4 Load Balancer (Transport Layer - TCP/UDP)**
    - Routes traffic based on IP address and port.
    - Example: **AWS Network Load Balancer (NLB)**
2. **Layer 7 Load Balancer (Application Layer - HTTP/HTTPS)**
    - Routes traffic based on URL, headers, cookies, etc.
    - Example: **AWS Application Load Balancer (ALB)**

### **Key Features**

✅ **Traffic Distribution** – Prevents overloading a single server

✅ **Health Checks** – Removes unhealthy instances from the pool

✅ **Session Persistence** – Maintains user sessions

✅ **SSL Termination** – Handles encryption and decryption for HTTPS

### **Example**

- **AWS Elastic Load Balancer (ELB)**
- **Azure Load Balancer**
- **Google Cloud Load Balancer**

### **Use Case**

- Scaling **web applications** horizontally
- Ensuring **zero downtime** by distributing traffic across multiple servers
- Redirecting traffic based on **geolocation or request type**

---

## **3. Security Groups**

### **Definition**

A **Security Group** functions as a **virtual firewall** that controls **inbound and outbound traffic** for cloud resources.

### **Key Features**

✅ **Stateful** – When an inbound rule allows traffic, the response is automatically allowed

✅ **Inbound & Outbound Rules** – Defines allowed sources and destinations

✅ **Instance-Level Security** – Attached to specific VMs or services

### **Example**

- **AWS Security Groups**
- **Azure Network Security Groups (NSG)**
- **Google Cloud Firewall Rules**

### **Use Case**

- Restricting **SSH (port 22) access** to specific IPs
- Allowing **only HTTP/HTTPS traffic (ports 80 & 443)** to web servers
- Blocking **all outbound traffic except specific ports** for database security

---

### **Comparison Table**

| Feature | Virtual Network | Load Balancer | Security Group |
| --- | --- | --- | --- |
| **Purpose** | Provides private networking | Distributes traffic | Controls access rules |
| **Scope** | Network-wide | Application-wide | Instance-level |
| **Key Benefit** | Isolates & connects resources | Ensures availability | Enhances security |
| **Example Use Case** | Connecting multiple servers | Scaling web apps | Restricting unauthorized access |

---

### **How They Work Together**

1. **A Virtual Network (VNet/VPC)** hosts cloud resources such as web servers and databases.
2. **A Load Balancer** distributes traffic across multiple servers inside the virtual network.
3. **Security Groups** enforce access control by allowing only specific types of traffic.

---

### **Example Scenario: Deploying a Scalable Web Application**

1. **Virtual Network** – Creates a private network with **subnets** for application and database layers.
2. **Load Balancer** – Distributes incoming traffic across **multiple application servers**.
3. **Security Groups** – Allows **only HTTP/HTTPS** traffic from users and restricts **database access**.

# Containers

A **container** is a lightweight, portable, and self-sufficient unit that includes an application along with its dependencies, libraries, and runtime environment. It ensures that the application runs consistently across different environments (development, testing, and production) without worrying about differences in system configurations.

- Containers are isolated from each other and the host system.
- They are more efficient than virtual machines because they share the same OS kernel.
- Common containerization technologies: Docker, Podman, LXC.

## **Docker**

Docker is a popular **containerization platform** that enables developers to package applications into containers. It provides tools to create, manage, and distribute containers efficiently.

Key Docker components:

- **Docker Engine** – Runs and manages containers.
- **Docker Image** – A lightweight, portable, and immutable file that contains application code and dependencies.
- **Docker Container** – A running instance of a Docker image.
- **Docker Hub** – A cloud-based registry for storing and sharing Docker images.
- **Docker Compose** – A tool to define and manage multi-container applications.

## **Kubernetes**

Kubernetes (K8s) is an **open-source container orchestration platform** that automates the deployment, scaling, and management of containerized applications across multiple nodes.

Key Kubernetes components:

- **Pod** – The smallest deployable unit in Kubernetes, consisting of one or more containers.
- **Node** – A machine (VM or physical) that runs Kubernetes-managed workloads.
- **Cluster** – A set of nodes controlled by a Kubernetes master.
- **Deployment** – Defines how an application should run and scale.
- **Service** – Exposes an application running on a set of Pods to the network.

## **How They Work Together**

- **Docker** is used to create and manage containers.
- **Kubernetes** orchestrates and manages multiple containers at scale.
- Kubernetes can run containers created by Docker or other container runtimes (like containerd or CRI-O).

## **Containers, Docker, and Kubernetes Explained with Examples**

### **1. Containers Example**

A container packages an application with all its dependencies to ensure consistent operation across different environments.

**Example:**

Consider a Python web application that uses Flask and specific system libraries. Rather than manually installing Flask and its dependencies on each machine, you can create a container with everything your app needs.

A simple **Dockerfile** for this Flask app:

```docker
# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the application
CMD ["python", "app.py"]
```

Now, you can **build and run the container**:

```bash
docker build -t flask-app .
docker run -p 5000:5000 flask-app
```

Your app runs inside a container, ready to share with anyone—no dependency issues to worry about.

---

### **2. Docker Example**

Docker simplifies container creation and management.

**Example: Running a MySQL database in a Docker container**

```bash
docker run --name my-mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -d mysql:latest
```

- `docker run` → Starts a new container.
- `-name my-mysql` → Names the container.
- `e MYSQL_ROOT_PASSWORD=my-secret-pw` → Sets an environment variable.
- `d mysql:latest` → Runs MySQL in the background.

Just like that, you have a MySQL database running in a container—no complex setup required.

---

### **3. Kubernetes Example**

Kubernetes is used to **orchestrate multiple containers** across a cluster.

Let's deploy a simple **Nginx web server** on Kubernetes.

**Step 1: Define a Kubernetes Deployment (nginx-deployment.yaml)**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        ports:
        - containerPort: 80
```

This configuration:

- Deploys **3 replicas** of an Nginx container.
- Uses the **nginx:latest** image.
- Exposes port **80** for web traffic.

**Step 2: Apply the Deployment**

```bash
kubectl apply -f nginx-deployment.yaml
```

Kubernetes will create and manage **three Nginx containers**.

**Step 3: Expose the Deployment as a Service**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: LoadBalancer
```

Now, users can access Nginx through a **LoadBalancer**.

**Step 4: Apply the Service**

```bash
kubectl apply -f nginx-service.yaml
```

Kubernetes ensures:

- The Nginx service runs **even if a container crashes**.
- The application scales **automatically**.

---

### **Summary**

| Feature | Containers | Docker | Kubernetes |
| --- | --- | --- | --- |
| Purpose | Isolate and run applications | Create & manage containers | Orchestrate multiple containers |
| Example | Flask app running inside a container | Running MySQL in Docker | Deploying Nginx on Kubernetes |
| Benefits | Portable, fast, consistent | Simplifies containerization | Scales and automates deployments |

# **Network and Subnet**

## **1. Network**

A **network** is a system that allows devices (computers, servers, routers, etc.) to communicate with each other by sending and receiving data. Networks can be **physical** (wired or wireless) or **virtual** (like cloud networks).

### **Types of Networks**

1. **Local Area Network (LAN)** – Connects devices within a small area (e.g., home, office).
2. **Wide Area Network (WAN)** – Connects multiple LANs across large distances (e.g., the Internet).
3. **Virtual Network (VPC/VNet)** – A cloud-based network that connects resources in a virtual environment.

---

## **2. Subnet (Subnetwork)**

A **subnet** is a **smaller segment** of a larger network, created to improve network efficiency, organization, and security. Subnetting divides a network into multiple **logical segments** to reduce congestion and optimize performance.

### **Why Use Subnets?**

✅ Improves **network performance** by reducing congestion

✅ Enhances **security** by isolating critical resources

✅ Allows **better IP address management**

✅ Enables **scalability** in large network infrastructures

---

## **Example of Network and Subnet**

Imagine a company with **one main network (10.0.0.0/16)** and different departments:

- **HR Subnet** → 10.0.1.0/24
- **IT Subnet** → 10.0.2.0/24
- **Finance Subnet** → 10.0.3.0/24

Each subnet isolates traffic within its department, improving security and performance.

---

## **Subnet Mask**

A **subnet mask** defines the network and host portions of an IP address.

| CIDR Notation | Subnet Mask | No. of Hosts |
| --- | --- | --- |
| `/8` | `255.0.0.0` | 16 million+ |
| `/16` | `255.255.0.0` | 65,536 |
| `/24` | `255.255.255.0` | 256 |

Example:

- IP Address: **192.168.1.10**
- Subnet Mask: **255.255.255.0 (/24)**
- Network: **192.168.1.0**
- Available Hosts: **192.168.1.1 - 192.168.1.254**

---

## **Comparison: Network vs. Subnet**

| Feature | Network | Subnet |
| --- | --- | --- |
| **Definition** | A group of interconnected devices | A smaller division of a network |
| **Purpose** | Enables communication | Organizes and optimizes network traffic |
| **Example** | `10.0.0.0/16` (entire network) | `10.0.1.0/24` (subnet within the network) |
| **Key Benefit** | Global connectivity | Security, efficiency, and scalability |