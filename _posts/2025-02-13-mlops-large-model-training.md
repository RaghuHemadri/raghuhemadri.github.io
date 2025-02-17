---
layout: post
title: "MLOps: Model Trainig at Scale"
date: 2025-02-13 09:00:00
description: Techniques for training large-scale machine learning models.
tags: mlops
categories: ml
thumbnail: assets/img/multi-gpu.webp
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

Training large-scale machine learning models presents challenges in computational efficiency, memory usage, and training speed. Modern deep learning models have millions to billions of parameters, requiring specialized techniques for effective training. In this blog post, we will explore foundational concepts, practical implementations, and hands-on techniques for training models that exceed the memory limits of a single GPU, as well as methods for optimizing training across multiple GPUs.

## Understanding Backpropagation and Its Challenges
Backpropagation is the fundamental algorithm for training neural networks. It consists of two phases:

1. **Forward Pass**: Data propagates through the network layer by layer, generating activations and predictions.
2. **Backward Pass**: Gradients are computed using the chain rule and propagated backward to update model parameters.

### Key Components in Backpropagation
- **Activations**: Outputs from each layer that serve as inputs to the next layer.
- **Gradients**: Partial derivatives of the loss function with respect to model parameters.
- **Weight Updates**: Adjustments to model parameters based on computed gradients.

Backpropagation relies on caching activations from the forward pass and using them during the backward pass to compute gradients efficiently. The larger the model, the more memory and computation are required.

### Scaling Backpropagation
Different aspects of training scale with:
- **Parameters**: Increasing the number of trainable parameters increases computation and memory consumption.
- **Activations**: Stored activations scale with batch size and depth, impacting memory usage.
- **Batch Size**: Larger batches improve statistical efficiency but require more memory.

## Training Large Models on a Single GPU
When a model does not fit into the memory of a single GPU, several techniques can be employed to optimize training:

### 1. Gradient Accumulation
Gradient accumulation allows us to simulate training with a larger batch size by accumulating gradients over multiple mini-batches before updating model weights. This is particularly useful when GPU memory is limited.

#### Implementation in PyTorch:
```python
scaler = torch.cuda.amp.GradScaler()
optimizer.zero_grad()
accumulation_steps = 4  # Adjust based on memory constraints

for i, (inputs, labels) in enumerate(dataloader):
    with torch.cuda.amp.autocast():  # Mixed precision for efficiency
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps  # Scale loss
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 2. Reduced and Mixed Precision Training
Reduced precision training decreases memory usage and speeds up computation by storing weights and activations in lower precision formats. 

#### Types of Reduced Precision:
- **Float16 (fp16)**: Uses 16-bit floating point, requiring loss scaling.
- **BFloat16 (bfp16)**: Provides better numerical stability with an 8-bit exponent.
- **Mixed Precision Training**: Combines full-precision (float32) and half-precision (float16) computations for efficiency.

#### Mixed Precision Training in PyTorch:
```python
scaler = torch.cuda.amp.GradScaler()

for inputs, labels in dataloader:
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Parameter-Efficient Fine-Tuning (LoRA & QLoRA)
Fine-tuning large models is expensive. Techniques such as **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** update only a subset of parameters, significantly reducing memory requirements.

- **LoRA** injects trainable low-rank matrices (A and B) into frozen model layers, efficiently learning task-specific adaptations.
- **QLoRA** quantizes pre-trained weights to 4-bit precision, further reducing memory usage.

#### LoRA Implementation with Hugging Face:
```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias='none')
model = get_peft_model(base_model, config)
```

## Strategies for Training Across Multiple GPUs
For extremely large models, distributed training is essential. The following techniques enable multi-GPU training:

### 1. Distributed Data Parallelism (DDP)
- **Each GPU holds a full copy of the model** but processes different slices of data.
- Gradients are computed locally and synchronized across GPUs.

#### DDP Implementation in PyTorch:
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("nccl")
model = DDP(model.to(device))
```

### 2. Fully Sharded Data Parallelism (FSDP)
FSDP shards model weights across GPUs instead of replicating them entirely. This allows training even larger models.

#### FSDP Implementation:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

fsdp_model = FSDP(model)
```

### 3. Model Parallelism (Tensor and Pipeline Parallelism)
For models too large for data parallelism, **model parallelism** splits computation across GPUs:

- **Tensor Parallelism**: Each GPU computes part of a layer’s matrix operations.
- **Pipeline Parallelism**: Different GPUs process different layers in a sequential pipeline.

#### Pipeline Parallelism Implementation:
```python
from torch.distributed.pipeline.sync import Pipe
segments = torch.nn.Sequential(model.layer1, model.layer2)
pipeline = Pipe(segments, chunks=8)
```

## Fully Sharded Data Parallelism (FSDP) in Depth
FSDP allows for efficient training of models larger than the memory of a single GPU by sharding parameters, optimizer states, and gradients.

1. **Forward Pass:** Each GPU loads only the necessary model weights for computation.
2. **Backward Pass:** Activations are stored and discarded layer by layer to save memory.
3. **Gradient Synchronization:** Uses Reduce-Scatter for efficiency.

## Summary and Future Directions
### Techniques for Single GPU Training:
✅ Gradient Accumulation
✅ Reduced Precision & Mixed Precision
✅ LoRA & QLoRA

### Techniques for Multi-GPU Training:
✅ Distributed Data Parallelism (DDP)
✅ Fully Sharded Data Parallelism (FSDP)
✅ Model Parallelism (Tensor and Pipeline)

These methods enable training beyond a single GPU, improving efficiency and reducing hardware constraints. In future posts, we will explore how these techniques can be packaged into scalable training services for organizations.


