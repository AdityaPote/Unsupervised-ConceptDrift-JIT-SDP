# Unsupervised-ConceptDrift-JIT-SDP

Just-in-Time Software Defect Prediction (JIT-SDP) is a critical task in software engineering aimed at identifying defect-prone code changes at the time of their submission. This project integrates an **unsupervised concept drift detection framework** with a JIT-SDP model to enhance its robustness and adaptability to dynamic and evolving data distributions, ensuring reliable defect prediction over time.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Evaluation and Results](#evaluation-and-results)
7. [Contributing](#contributing)

---

## Introduction

Modern software development involves rapidly changing environments, often causing traditional defect prediction models to degrade in performance due to **concept drift** — changes in the underlying data distribution over time. This project:
- Enhances JIT-SDP using **unsupervised concept drift detection**.
- Adapts models in real-time to evolving data without relying on labeled data.
- Combines deep learning embeddings with statistical drift detection techniques to maintain prediction reliability.

The implemented framework leverages techniques to detect and characterize data drift while ensuring real-time adaptability for software defect prediction.

---

## Features

- **Concept Drift Detection**: Detects and responds to changes in data distribution using unsupervised statistical methods.
- **Integration with JIT-SDP Models**: Improves model adaptability and robustness.
- **Scalable and Efficient**: Processes data streams in real-time with low computational overhead.
- **Open Source**: The complete codebase is available for community use and improvement.

---

## Installation

To set up and run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/AdityaPote/Unsupervised-ConceptDrift-JIT-SDP.git
   cd Unsupervised-ConceptDrift-JIT-SDP

2 Create a Python virtual environment

    python -m venv env
    source env/bin/activate  # For Linux/macOS
    env\Scripts\activate     # For Windows

3 Install dependencies

    pip install -r requirements.txt


