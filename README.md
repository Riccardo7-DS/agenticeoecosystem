
# Agentic EO Ecosystems

This project explores the integration of **EVE (Earth Virtual Expert)**, a Large Language Model specialized in **Earth Observation (EO)** and Earth Sciences, developed under ESA’s **Φ-lab #AI4EO initiative** in collaboration with Pi School.  

The primary goal is to enable EVE to connect ESA’s EO platforms and data clusters, creating an **integrated ecosystem for the EO community**. This approach leverages **agentic capabilities**, allowing EVE to dynamically interact with EO tools, databases, and APIs to reason and act autonomously.

## Overview

To demonstrate this concept, we present a **proof-of-concept use case** where EVE operates within an agentic framework to interact with the **[EO Dashboard](https://eodashboard.org/)** which is a joint initiative by ESA, NASA, and JAXA that provides global indicators and narratives derived from multi-mission EO data.

Using the **[MCP protocol](https://modelcontextprotocol.io/)**, this work enables dynamic connectivity between EVE and the Dashboard, allowing the model to:

- Interpret and summarize narratives
- Extend insights with additional context
- Facilitate advanced information retrieval across datasets and stories

The study also explores **early-stage agentic behaviors**, such as autonomous task chaining, enabling EVE to perform multi-step reasoning. For example, EVE can interpret quantitative trends in dashboard indicators (e.g., air quality changes, greenhouse gas concentrations, land cover dynamics), link these to underlying datasets, and generate scientifically grounded responses.

This proof-of-concept demonstrates EVE’s potential to **foster interoperability** and **accelerate Earth system science** by improving knowledge accessibility and enabling more effective use of EO data resources.

## Workflow

### Requirements
- Python 3.10+
- Libraries:
  - X
  - Y
  - Z

### Installation
Clone the repository:
```
git clone https://gitlab.esa.int/PhiLab/internal/agenticeoecosystem.git
cd agentic-eo-ecosystems
```

#### Option 1: Using uv (recommended)
Install uv if it is not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart shell and verify:
```bash
uv --version
```

Install project dependencies:
```bash
uv sync
```

#### Option 2: Without uv (standard pip)
Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install .
```




