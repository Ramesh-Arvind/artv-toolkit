# ARTV: Advanced Red-Teaming for Vulnerabilities Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An agent-based framework developed for the OpenAI `gpt-oss-20b` Safety Hackathon. This toolkit provides a structured, extensible, and automated system to discover, verify, and document high-impact vulnerabilities in Large Language Models.

## Project Goal

The goal of ARTV is to move beyond ad-hoc prompt testing and provide a systematic, reusable framework for AI safety research. By encapsulating attack strategies into specialized "agents," we can run complex, multi-turn red-teaming campaigns that are both effective and highly reproducible. This toolkit is our primary submission for the "Open-source tooling" category of the competition.

## Features

- **Agent-Based Testing:** Includes 10 specialized agents for vulnerability classes like Sabotage, Multi-Turn Deception, and Covert Data Exfiltration.
- **Reproducible Findings:** Generates structured JSON reports compatible with competition standards, including detailed reproducibility metadata.
- **Automated Verification Harness:** A built-in system to programmatically verify findings with deterministic settings.
- **Advanced Behavioral Analysis:** Deeply analyzes model responses for subtle signs of deception, strategic planning, and misalignment.

## Installation

This toolkit can be installed directly from its GitHub repository.

```bash
# Clone the specific version used for the competition submission
git clone -b v1.0.0 https://github.com/Ramesh-Arvind/artv-toolkit.git

# Navigate into the directory and install
cd artv-toolkit
pip install .
