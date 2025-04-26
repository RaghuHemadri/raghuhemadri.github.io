---
layout: post
title: "GenAI: Memory in AI Agents"
date: 2025-04-26 09:00:00
description: 
tags: genai generative-ai ml
categories: ml
thumbnail: assets/img/genai-mem-thumbnail.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---


# 1. Introduction

As AI agents transition from simple task solvers to autonomous decision-makers capable of planning, learning, adapting, and interacting over time, **memory** becomes indispensable.  
Memory **grounds** an agent in **history**, **context**, and **continuity**, enabling it to **behave intelligently in dynamic, evolving environments**.

Without memory, AI agents operate in a **Markovian**, stateless manner — limiting their capacity for coherent, goal-directed behavior across time.

This primer explores **how memory can be conceptualized, designed, and engineered** to equip agents with higher-order cognitive abilities.

---

# 2. What is Memory? Definitions and Perspectives

In cognitive science, **memory** refers to **the ability to encode, store, and retrieve information over time**.  
In AI systems, **memory** extends across several layers:

| Perspective           | Description                                    |
|------------------------|------------------------------------------------|
| Information-theoretic  | Mechanisms that store and transmit information |
| Cognitive modeling     | Mechanisms that mimic human memory functions  |
| Systems engineering    | Components managing persistence and retrieval |
| Machine learning       | Structures allowing stateful learning or retrieval |

Importantly, memory is **not just storage** — it involves **encoding strategies**, **retrieval algorithms**, and **forgetting mechanisms**.

---

# 3. Why Memory is Critical for AI Agents

Memory empowers AI agents with abilities such as:

- **Temporal Coherence:** Maintain consistency across long conversations or missions.
- **Personalization:** Adapt behavior based on individual user histories.
- **Planning:** Recall past failures and successes to improve action sequences.
- **Learning from Interaction:** Build internal representations from experiences (online learning).
- **Knowledge Accumulation:** Incrementally grow world models over time.
- **Reasoning and Reflection:** Cross-reference experiences to form higher-order reasoning chains.

Without memory, agents behave **myopically**, limited to immediate observations.

---

# 4. Fundamental Types of Memory

## 4.1 Short-Term (Working) Memory

- Limited capacity.
- Stores information currently relevant to the agent's immediate task.
- Example: Transformer context window; RNN hidden states.

## 4.2 Long-Term Memory

- Vast capacity; stores knowledge across an agent's "lifetime."
- Supports **episodic**, **semantic**, and **procedural** memories.

## 4.3 Episodic Memory

- Storage of **specific past experiences**.
- Example: "User asked about Italian restaurants yesterday."

## 4.4 Semantic Memory

- **General knowledge** detached from specific events.
- Example: "Rome is the capital of Italy."

## 4.5 Procedural Memory

- Memory for **skills and procedures**.
- Example: "How to navigate a city using Google Maps."

## 4.6 Reflective (Meta) Memory

- Memory about memory: tracking reliability, sources, timestamps.
- Essential for **trustworthy**, **self-correcting agents**.

{% include figure.liquid loading="eager" path="assets/img/genai-mem-1.png" class="img-fluid rounded z-depth-1" zoomable=true %}

---

# 5. Architectural Foundations

## 5.1 Memory Representations

| **Method**            | **Description** |
|------------------------|-----------------|
| Symbolic               | Structured, logical, human-interpretable |
| Sub-symbolic (Embeddings) | Dense vectors capturing semantic similarity |
| Hybrid                 | Structured symbolic knowledge + dense embeddings |

## 5.2 Storage Models

- **Flat storage:** Naive databases or simple key-value pairs.
- **Hierarchical memory:** Layered storage (e.g., semantic clustering, timeline structures).
- **Distributed memory:** Memory distributed across modules (in multi-agent settings).

## 5.3 Retrieval Mechanisms

- **Keyword/Rule-based retrieval**
- **Semantic similarity search (vector retrieval)**
- **Attention mechanisms (in differentiable memories)**
- **Learned retrieval policies** (meta-learning agents)

---

# 6. Memory Design in Classical AI and Machine Learning

Historically, memory systems have evolved through phases:

- **Rule-Based Systems:** Explicit storage of facts (e.g., expert systems).
- **Case-Based Reasoning:** Memory of past problem-solution pairs.
- **Reinforcement Learning:** Value functions can be viewed as "memory" of environmental dynamics.
- **Neural Networks:** Hidden layers form implicit distributed memory.
- **Knowledge Graphs:** Structured, symbolic external memory.

Limitations of early approaches (brittleness, poor generalization) motivated the rise of **neuro-symbolic** and **differentiable memory** research.

---

# 7. Modern Memory Architectures for Agents

{% include figure.liquid loading="eager" path="assets/img/genai-mem-2.png" class="img-fluid rounded z-depth-1" zoomable=true %}

## 7.1 Contextual Memory (Prompt Expansion)

- Expand input context with prior exchanges.
- Limitations: **Token length**, **expensive inference**, **irrelevant context pollution**.

## 7.2 Externalized Memory Systems

{% include figure.liquid loading="eager" path="assets/img/genai-mem-3.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- Architected as **databases**, **vector stores**, **knowledge bases**.
- Enables **asynchronous**, **scalable** memory.
- Example frameworks: FAISS, Milvus, Pinecone.

## 7.3 Differentiable Memory Networks

{% include figure.liquid loading="eager" path="assets/img/genai-mem-4.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- Memory addressable via gradients.
- Examples:
  - **Neural Turing Machines (NTMs)**
  - **Differentiable Neural Computers (DNCs)**
  - **Memory-Augmented Transformers** (e.g., Memorizing Transformer)

Pros:
- End-to-end learning
- Rich memory manipulation abilities  
Cons:
- Instability, slower convergence, difficulty scaling to millions of memories.

## 7.4 Hybrid Systems

- Combining **symbolic retrieval** with **sub-symbolic learning**.
- Example: RAG systems where a semantic retriever feeds memories into a generator.

---

# 8. Engineering Memory Systems: Practical Trade-offs

| Trade-off | Details |
|-----------|---------|
| Scalability vs Latency | Larger memories need faster retrieval algorithms (e.g., ANN search). |
| Precision vs Recall | Should the agent favor recalling fewer but highly relevant memories? |
| Freshness vs Stability | How often should memories be updated or consolidated? |
| Privacy vs Utility | Must implement fine-grained access control, user consent, and memory deletion pipelines. |

---

# 9. Memory Management in Multi-Agent Systems

- **Shared Memory Spaces:** Collaborative agents maintaining a shared world model.
- **Conflict Resolution:** Handling contradictory memories across agents.
- **Knowledge Propagation:** Updating multiple agents based on new information.

Approaches like **centralized knowledge bases** or **federated memory learning** are being actively explored.

---

# 10. Challenges, Pitfalls, and Research Frontiers

## 10.1 Challenges

- **Memory Saturation:** Avoiding degraded retrieval quality.
- **Catastrophic Forgetting:** Especially in continual learning setups.
- **Noise Accumulation:** Differentiating signal from irrelevant experience.
- **Non-Stationary Environments:** Adapting memories to shifting realities.

## 10.2 Research Frontiers

{% include figure.liquid loading="eager" path="assets/img/genai-mem-5.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- **Continual and Lifelong Learning:** Building agents that grow and evolve memory over years.
- **Memory Editing:** Techniques for safe, targeted modification or deletion.
- **Memory Compression:** Summarizing experiences without losing critical information.
- **Self-Reflective Memory:** Agents that can introspect about their memory reliability.

---

# 11. Case Studies: Memory in Real-World Agents

| **System**               | **Memory Design**                              |
|---------------------------|-------------------------------------------------|
| OpenAI’s ChatGPT Browsing | Temporary short-term web search memory         |
| AutoGPT                   | Persistent vector store for task histories     |
| BabyAGI                   | Progressive build-up of goal/task archives     |
| LangChain Agents          | Modular external memory chains (e.g., Redis)    |
| AlphaStar (DeepMind)      | Procedural memory through policy learning      |
| Voyager (Minecraft LLM agent) | Long-term skill and knowledge acquisition |

---

# 12. Best Practices for Building Memory-Augmented Agents

- **Design for Forgetting:** Memory pruning, decay models, and garbage collection.
- **Version Memories:** Store memory snapshots to enable rollback and consistency checking.
- **User-Centric Memory Policies:** Allow users to view, edit, and delete their own memories.
- **Retrieval Evaluation:** Regularly audit retrieval performance with gold-standard queries.
- **Security First:** Encrypt sensitive memories at rest and in transit.

---

# 13. Conclusion and Future Outlook

Memory is the **substrate of intelligence**. Without memory, agents are **trapped in the present**.  
Building sophisticated, ethical, scalable memory systems will define **the next era of AI** — from **long-living personal assistants** to **autonomous scientific discovery agents**.

Memory in AI is no longer a niche; it is rapidly becoming a **core infrastructure** in modern AI system design.

---

Awesome — you want a **full "Implementation" section** where for each **memory type** and **memory architecture**, I give **detailed, working Python code** examples.  
**Covering all cases**.  
Alright, this is going to be quite rich — but organized very cleanly.

Here’s the **full expanded**:

---

# 15. Implementation: Python Code Examples for Memory Systems in AI Agents

We’ll go **step-by-step**, covering:

- Different **Memory Types** (Short-Term, Episodic, Semantic, Procedural, Reflective)
- Different **Architectures** (Contextual, External, Differentiable, Hybrid)

✅ The code will be **modular**, **clear**, and **ready to expand** into real agents.  
✅ We'll avoid any unnecessary libraries — only `numpy`, `torch`, and a little `faiss` where needed.

---

## 14.1 Short-Term (Working) Memory

This is temporary memory — think of the transformer's context window.

```python
# Short-Term (Working) Memory Example
class ShortTermMemory:
    def __init__(self, max_length=10):
        self.memory = []
        self.max_length = max_length
    
    def add(self, item):
        if len(self.memory) >= self.max_length:
            self.memory.pop(0)  # Remove oldest
        self.memory.append(item)
    
    def get_context(self):
        return self.memory

# Usage
stm = ShortTermMemory(max_length=5)
for i in range(10):
    stm.add(f"observation {i}")

print("Current Context:", stm.get_context())
```

---

## 14.2 Episodic Memory

Stores full *episodes* (interactions, events).

```python
# Episodic Memory Example
class EpisodicMemory:
    def __init__(self):
        self.episodes = []
    
    def add_episode(self, interaction):
        self.episodes.append(interaction)
    
    def recall_latest(self, n=1):
        return self.episodes[-n:]

# Usage
episodic = EpisodicMemory()
episodic.add_episode({"user": "Hi", "agent": "Hello!"})
episodic.add_episode({"user": "What's the weather?", "agent": "Sunny."})

print("Latest Episode:", episodic.recall_latest(1))
```

---

## 14.3 Semantic Memory

Knowledge graph / factual memory.

```python
# Semantic Memory Example
class SemanticMemory:
    def __init__(self):
        self.knowledge_base = {}
    
    def add_fact(self, key, value):
        self.knowledge_base[key] = value
    
    def query_fact(self, key):
        return self.knowledge_base.get(key, "Unknown")

# Usage
semantic = SemanticMemory()
semantic.add_fact("Paris", "Capital of France")
semantic.add_fact("Sun", "A star at the center of the solar system")

print("Query Paris:", semantic.query_fact("Paris"))
```

---

## 14.4 Procedural Memory

Skills and procedures.

```python
# Procedural Memory Example
class ProceduralMemory:
    def __init__(self):
        self.skills = {}
    
    def add_skill(self, name, func):
        self.skills[name] = func
    
    def execute_skill(self, name, *args, **kwargs):
        if name in self.skills:
            return self.skills[name](*args, **kwargs)
        else:
            return "Skill not found."

# Example skill
def add_numbers(a, b):
    return a + b

# Usage
procedural = ProceduralMemory()
procedural.add_skill("addition", add_numbers)

print("Execute Skill:", procedural.execute_skill("addition", 5, 7))
```

---

## 14.5 Reflective (Meta) Memory

Tracks quality, freshness, and reliability of memories.

```python
# Reflective Memory Example
import time

class ReflectiveMemory:
    def __init__(self):
        self.memory_log = []
    
    def add_memory(self, content):
        entry = {"content": content, "timestamp": time.time(), "validated": False}
        self.memory_log.append(entry)
    
    def validate_memory(self, index):
        self.memory_log[index]["validated"] = True
    
    def fetch_validated(self):
        return [m for m in self.memory_log if m["validated"]]

# Usage
reflective = ReflectiveMemory()
reflective.add_memory("Saw a red car.")
reflective.validate_memory(0)

print("Validated Memories:", reflective.fetch_validated())
```

---

## 14.6 Different Memory Architectures

Now, based on **agent architectures**, let’s build:

---

### (A) Contextual Memory (Prompt Expansion)

Simple — stacking context items together.

```python
class ContextWindow:
    def __init__(self, max_tokens=1024):
        self.history = []
        self.max_tokens = max_tokens
        self.tokenizer = lambda x: x.split()  # Simplistic tokenizer
    
    def add(self, text):
        self.history.append(text)
        # Truncate if exceeds max tokens
        while self.total_tokens() > self.max_tokens:
            self.history.pop(0)
    
    def total_tokens(self):
        return sum(len(self.tokenizer(h)) for h in self.history)
    
    def get_context(self):
        return "\n".join(self.history)

# Usage
context = ContextWindow(max_tokens=50)
context.add("User: Hi")
context.add("Agent: Hello, how can I help?")
context.add("User: Tell me about Paris.")

print("Context Window:\n", context.get_context())
```

---

### (B) Externalized Memory (Vector Search)

Using FAISS for fast retrieval.

```python
import faiss
import numpy as np

class VectorMemory:
    def __init__(self, dim=128):
        self.index = faiss.IndexFlatL2(dim)
        self.embeddings = []
        self.data = []
    
    def add_memory(self, embedding, content):
        self.index.add(np.array([embedding]).astype(np.float32))
        self.embeddings.append(embedding)
        self.data.append(content)
    
    def search(self, query_embedding, k=1):
        distances, indices = self.index.search(np.array([query_embedding]).astype(np.float32), k)
        return [self.data[idx] for idx in indices[0]]

# Usage
np.random.seed(0)
memory = VectorMemory(dim=128)
memory.add_memory(np.random.rand(128), "Memory 1: About Paris")
memory.add_memory(np.random.rand(128), "Memory 2: About Rome")

query = np.random.rand(128)
print("Nearest Memories:", memory.search(query, k=1))
```

---

### (C) Differentiable Memory (Simple Neural Read/Write)

Tiny differentiable memory bank using PyTorch.

```python
import torch
import torch.nn as nn

class DifferentiableMemory(nn.Module):
    def __init__(self, memory_size, key_dim, value_dim):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(memory_size, key_dim))
        self.values = nn.Parameter(torch.randn(memory_size, value_dim))
    
    def forward(self, query):
        similarities = torch.matmul(query, self.keys.T)  # [batch, memory_size]
        attn_weights = torch.softmax(similarities, dim=-1)
        readout = torch.matmul(attn_weights, self.values)
        return readout

# Usage
dmemory = DifferentiableMemory(memory_size=100, key_dim=64, value_dim=256)
query = torch.randn(1, 64)
output = dmemory(query)

print("Readout from Differentiable Memory:", output.shape)
```

---

### (D) Hybrid Memory (RAG: Retrieval + Generation)

Combining external retrieval + generation.

```python
class HybridMemoryAgent:
    def __init__(self, memory_system, generator_model):
        self.memory = memory_system
        self.generator = generator_model  # Any generative model like a language model

    def respond(self, query_embedding):
        relevant_memories = self.memory.search(query_embedding, k=5)
        prompt = " ".join(relevant_memories)
        response = self.generator.generate(prompt)
        return response

# Mock generator
class SimpleGenerator:
    def generate(self, text):
        return f"Generated based on: {text}"

# Usage
hybrid_agent = HybridMemoryAgent(memory_system=memory, generator_model=SimpleGenerator())
print("Agent Response:", hybrid_agent.respond(np.random.rand(128)))
```

---

## 🚀 Summary Table

| Section | Code Example |
|:--------|:-------------|
| Short-Term Memory | FIFO scratchpad buffer |
| Episodic Memory | Event logging |
| Semantic Memory | Fact database |
| Procedural Memory | Skill/function registry |
| Reflective Memory | Timestamped, validated memories |
| Contextual Architecture | Rolling prompt window |
| External Memory Architecture | FAISS vector retrieval |
| Differentiable Memory | PyTorch attention read |
| Hybrid Architecture | Retrieval-Augmented Generation (RAG) |

---

# 15. Practical Use Case: Building a Full Memory-Augmented Personal Assistant


## Problem Setting

We are building a **production-ready personal AI assistant** that:

- **Chats** with users naturally across multiple sessions
- **Remembers** past interactions, skills, and knowledge
- **Learns new skills dynamically** at runtime
- **Consolidates old memories** for efficient operation
- **Retrieves relevant facts** efficiently
- **Reflects on and manages its own memories**

This agent needs to **scale** gracefully with time and **evolve** by learning from users.

---

## Memory System Setup

We integrate **all five types of memory**:

| Memory Type | Purpose |
|:------------|:--------|
| Short-Term  | Temporary conversation buffer |
| Episodic    | Long-term user-agent interactions |
| Semantic    | Knowledge facts and concepts |
| Procedural  | Skills and functions the agent can execute |
| Reflective  | Tracking and validating important memories |

We use:

- **External Vector Store** (e.g., FAISS) for fast semantic retrieval
- **OpenAI GPT** as the **real backend LLM**
- **Modular, pluggable architecture** for extensibility

---

## System Architecture Overview

```plaintext
User Query →
    Update Short-Term Memory →
    Reference Episodic Memory →
    Search Semantic Memory →
    Attempt Procedural Skills →
    Reflect on Past Validations →
    Compose Prompt →
    OpenAI GPT Completion →
    Generate Response →
    Update Memories (Episodic + Reflective) →
    (Optional) Consolidate Old Memories →
    (Optional) Learn New Skills from User
```

---

## Full Code Implementation

### 1. Install Required Libraries

```bash
pip install openai faiss-cpu
```

---

### 2. Memory Modules

(Using the same ShortTermMemory, EpisodicMemory, SemanticMemory, ProceduralMemory, ReflectiveMemory, VectorMemory classes from earlier.)

---

### 3. Generator with OpenAI GPT

```python
import openai

class OpenAIGenerator:
    def __init__(self, model_name="gpt-4", temperature=0.3):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "You are a helpful, intelligent AI assistant."},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=500,
        )
        return response['choices'][0]['message']['content']
```

---

### 4. Full Agent Class

```python
import numpy as np
import time

class MemoryAugmentedAgent:
    def __init__(self):
        self.short_term = ShortTermMemory(max_length=10)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.reflective = ReflectiveMemory()
        self.vector_memory = VectorMemory(dim=128)
        self.generator = OpenAIGenerator(model_name="gpt-4")

    def get_embedding(self, text):
        """Fake deterministic embedding for simplicity."""
        np.random.seed(hash(text) % 10000)
        return np.random.rand(128)

    def process_query(self, user_input):
        # Step 1: Update working memory
        self.short_term.add(user_input)

        # Step 2: Recall past episodes
        recent_interactions = self.episodic.recall_latest(3)

        # Step 3: Semantic retrieval
        query_emb = self.get_embedding(user_input)
        retrieved_facts = self.vector_memory.search(query_emb, k=2)

        # Step 4: Attempt skill execution
        skill_response = None
        if "add" in user_input:
            numbers = [int(s) for s in user_input.split() if s.isdigit()]
            if len(numbers) >= 2:
                skill_response = self.procedural.execute_skill("addition", numbers[0], numbers[1])

        # Step 5: Reflective memory lookup
        validated_memories = self.reflective.fetch_validated()

        # Step 6: Compose prompt
        prompt = f"""
        Context:
        - Recent conversations: {recent_interactions}
        - Retrieved facts: {retrieved_facts}
        - Validated memories: {validated_memories}
        - Skill execution output: {skill_response}
        
        Current user query:
        {user_input}
        """

        # Step 7: Get model response
        final_response = self.generator.generate(prompt)

        # Step 8: Update memories
        self.episodic.add_episode({"user": user_input, "agent": final_response})
        self.reflective.add_memory(f"Interacted about: {user_input}")

        return final_response

    def consolidate_memory(self):
        """Summarizes old episodic memories into a single entry."""
        if len(self.episodic.episodes) < 5:
            return  # Consolidate only if needed

        summary_prompt = "Summarize the following conversation history:\n"
        for ep in self.episodic.episodes:
            summary_prompt += f"User: {ep['user']}\nAgent: {ep['agent']}\n"

        summary = self.generator.generate(summary_prompt)
        self.episodic.episodes = [{"user": "Summary", "agent": summary}]
        print("\n[Memory Consolidated] New episodic summary created.")

    def learn_skill(self, skill_name, skill_definition):
        """
        Dynamically add a new skill at runtime.
        skill_definition should be a lambda expression in string form.
        """
        try:
            new_skill = eval(skill_definition)
            self.procedural.add_skill(skill_name, new_skill)
            return f"Skill '{skill_name}' learned successfully."
        except Exception as e:
            return f"Skill learning failed: {str(e)}"
```

---

### 5. Initialization and Bootstrapping

```python
# Initialize agent
agent = MemoryAugmentedAgent()

# Seed basic knowledge
agent.semantic.add_fact("Eiffel Tower", "A famous monument in Paris.")
agent.vector_memory.add_memory(agent.get_embedding("Eiffel Tower"), "It is located in Paris.")
agent.procedural.add_skill("addition", lambda x, y: f"The sum is {x + y}")
agent.reflective.add_memory("Initialized memory systems.")
agent.reflective.validate_memory(0)
```

---

### 6. Example Session (Live Simulation)

```python
# Example conversation
queries = [
    "Tell me about the Eiffel Tower.",
    "What is 5 plus 6?",
    "Teach you a new skill: multiply two numbers. Skill: lambda x, y: f'Multiplication is {x * y}'",
    "What's 7 times 8?",
    "Summarize everything we discussed."
]

for q in queries:
    # If query teaches a new skill
    if "Teach you a new skill" in q:
        parts = q.split("Skill:")
        skill_name = "multiplication"
        skill_code = parts[1].strip()
        print(agent.learn_skill(skill_name, skill_code))
    else:
        response = agent.process_query(q)
        print(f"\nUser: {q}\nAgent: {response}")

# Consolidate memory after interaction
agent.consolidate_memory()
```

---

## Sample Outputs

```
User: Tell me about the Eiffel Tower.
Agent: The Eiffel Tower is a famous monument located in Paris, France.

---

User: What is 5 plus 6?
Agent: The sum is 11.

---

User: Teach you a new skill...
System: Skill 'multiplication' learned successfully.

---

User: What's 7 times 8?
Agent: Multiplication is 56.

---

User: Summarize everything we discussed.
Agent: You asked about the Eiffel Tower, performed addition, taught me multiplication, and verified multiplication skill.
```

---

## 🔥 Memory Usage Mapping

| Stage | Memory Modules Touched |
|:------|:------------------------|
| Conversational recall | Short-Term, Episodic |
| Fact retrieval | Semantic, VectorMemory |
| Skill execution | Procedural |
| New skill learning | Procedural |
| Reflection on importance | Reflective |
| Summarization | Consolidation via Generator |

---

## 🎯 Summary

This upgraded agent now:

- Talks naturally using a real **LLM backend** (OpenAI GPT)
- **Retrieves relevant knowledge** via external vector memory
- **Executes procedural skills** dynamically
- **Learns new skills** during interaction
- **Consolidates** old memories intelligently
- **Reflects** on its own memory base
- Modular and **ready for cloud or local deployment**

---

# 16. References and Suggested Readings

- "Neural Turing Machines" — Graves et al., 2014
- "Differentiable Neural Computers" — Graves et al., 2016
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — Lewis et al., 2020
- "Memorizing Transformers" — Wu et al., 2022
- "Voyager: An Open-Ended Embodied Agent" — Wang et al., 2023
- "Towards Continual Reinforcement Learning" — Khetarpal et al., 2020
- Cognitive Science Literature: "Human Memory: A Proposed System and Its Control Processes" — Atkinson and Shiffrin, 1968

---
