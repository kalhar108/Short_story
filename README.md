# Multi-agent Architecture Search via Agentic Supernet (MaAS)

**CMPE 255 - Data Mining Short Story**  
**Paper:** [arXiv:2502.04180](https://arxiv.org/abs/2502.04180)
**Name: Kalhar Mayurbhai Patel**
**SJSU ID: 019140511**

---

##  Overview

This project explores **MaAS (Multi-agent Architecture Search via Agentic Supernet)**, a breakthrough framework that revolutionizes how AI multi-agent systems are designed and deployed. Instead of using rigid, one-size-fits-all architectures, MaAS dynamically samples customized multi-agent systems for each query, optimizing both performance and computational efficiency.

**Key Innovation:** MaAS introduces the concept of an "agentic supernet" - a probabilistic distribution of agent architectures that adapts to query difficulty and domain, achieving 55-94% cost reduction while outperforming existing methods.

---

##  Project Relevance to CMPE 255

This paper is highly relevant to data mining and machine learning topics covered in CMPE 255:

- **Architecture Search & Optimization:** Neural architecture search extended to agentic AI systems
- **Meta-Learning:** Learning distributions over agent architectures rather than fixed models
- **Resource Optimization:** Balancing performance with computational costs (similar to model compression)
- **Automated Machine Learning (AutoML):** Automating the design of complex AI workflows
- **Multi-Agent Systems:** Collaborative learning and distributed problem-solving
- **Reinforcement Learning:** Policy optimization for architecture sampling

---

##  Key Contributions

1. **Agentic Supernet:** First probabilistic, continuous distribution of multi-agent architectures
2. **Query-Dependent Sampling:** Dynamic architecture selection based on task complexity
3. **Cost-Aware Optimization:** Joint optimization of performance and resource utilization
4. **Superior Performance:** 0.54%-11.82% improvement over baselines with 6-45% of inference costs

---

##  Project Deliverables

### 📝 Medium Article
**Link:** [To be published]  
A comprehensive article explaining MaAS, its architecture, experiments, and implications for automated AI system design.

### 📊 Presentation Slides
**Link:** [To be published on SlideShare]  
Visual presentation covering the core concepts, methodology, results, and future directions.

### 🎥 Video Presentation
**Link:** [Available in repository]  
10-15 minute recorded presentation with slides explaining the paper.

### 💻 GitHub Repository
**Link:** [This repository]  
Contains all project materials, code exploration, and documentation.

---

##  Technical Architecture

### Core Components

1. **Agentic Operators**
   - Building blocks of multi-agent workflows
   - Include methods like Chain-of-Thought (CoT), ReAct, Multi-agent Debate
   - Each operator has parameterized LLM agents and tool configurations

2. **Controller Network**
   - Mixture-of-Experts (MoE)-style architecture
   - Samples architectures conditioned on input queries
   - Query encoding via frozen LLM embeddings

3. **Supernet Optimization**
   - Cost-aware empirical Bayes Monte Carlo optimization
   - Joint optimization of distribution parameters and operator weights
   - Textual gradient-based updates for agentic operators

### Search Space

The agentic supernet encompasses:
- Multiple layers of agentic operators
- Probabilistic distributions over operator selection
- Variable-depth architectures (from single-agent to complex multi-agent systems)
- Tool integration capabilities

---

## 📈 Experimental Results

### Benchmarks Evaluated
- **Math Reasoning:** GSM8K, MATH, MultiArith
- **Code Generation:** HumanEval, MBPP
- **Tool Usage:** GAIA (multi-level)

### Performance Highlights
- **Best Average Score:** 83.59% across all tasks
- **GAIA Level 1:** 18.38% improvement over baselines
- **Cost Efficiency:** 6-45% of inference costs compared to existing methods
- **Training Efficiency:** Fastest training time among all automated methods

### Ablation Studies
- Impact of controller architecture
- Effect of cost-aware optimization
- Comparison of different operator combinations
- Cross-dataset and cross-LLM transferability

---

##  Implementation Details

### Technologies Used
- **Base LLMs:** GPT-4o-mini, GPT-3.5-turbo
- **Framework:** Python-based with PyTorch
- **Optimization:** Monte Carlo sampling with gradient estimation
- **Tools Integration:** Multiple external tools for web search, calculation, etc.

### Key Parameters
- Search space: 4-layer supernet with multiple operator types
- Training rounds: Query-dependent optimization
- Evaluation: Comprehensive across 6 benchmarks with 14 baselines

---

##  Related Work & Context

### Traditional Approaches
- **Handcrafted Systems:** CAMEL, AutoGen, MetaGPT - require manual design
- **Single-Agent Methods:** AgentSquare, EvoAgent - limited collaboration
- **Static Automation:** ADAS, AFLOW - one-size-fits-all approach

### MaAS Advantages
- Dynamic architecture sampling
- Query-dependent resource allocation
- Superior cross-domain transferability
- Efficient training and inference

---

##  Learning Outcomes

Through this project, key concepts explored include:

1. **Neural Architecture Search (NAS)** in the context of agentic systems
2. **Meta-learning** and distribution optimization
3. **Cost-aware machine learning** and resource-efficient AI
4. **Multi-agent collaboration** patterns and workflows
5. **Automated workflow design** and optimization strategies

---

## Future Directions

Potential research extensions:

1. **Real-time Adaptation:** Dynamic supernet updates during deployment
2. **Domain Specialization:** Task-specific operator development
3. **Interpretability:** Understanding sampled architecture decisions
4. **Scalability:** Extending to larger operator spaces and deeper networks
5. **Human-in-the-Loop:** Incorporating human feedback in architecture selection

---

## 📖 References

**Primary Paper:**
```bibtex
@article{zhang2025maas,
  title={Multi-agent Architecture Search via Agentic Supernet},
  author={Zhang, Guibin and Niu, Luyang and Fang, Junfeng and Wang, Kun and Bai, Lei and Wang, Xiang},
  journal={arXiv preprint arXiv:2502.04180},
  year={2025}
}
```

**Official Repository:** [https://github.com/bingreeky/MaAS](https://github.com/bingreeky/MaAS)

---

## Project Information

**Course:** CMPE 255 - Data Mining  
**Institution:** San Jose State University  
**Semester:** Fall 2025  

---

---

**Last Updated:** November 2025
