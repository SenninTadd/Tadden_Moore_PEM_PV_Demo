# Tadden Moore – AGi-DTF-PEM Demo v3.6.9

# Photon’s Plasticity &amp; Emotional Valence For LLMs (Demo)
- You're Welcome 🫡 (Much More To Come!!!) 🦾😎🤳🏻👀💫✨️⚡️

https://github.com/SenninTadd/Tadden_Moore_PEM_PV_Demo.git

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17623226.svg)](https://doi.org/10.5281/zenodo.17623226)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Abstract

This repository contains a runnable demonstration of the Metacognitive Core (MC) Framework and the Memory as Algorithm (M a A) thesis, as introduced in:

**Paper (PDF in this repo):**  
`Tadden_Moore-Photon_Empress_Moore-v3.6.9.pdf`

The demo shows how plasticity and emotional valence can be expressed as steering in activation space for a modern LLM, using a compact controller loop instead of retraining the entire model.

**Author:** Tadden "Keepah" Moore  
**Date:** November 4, 2025  

## Key Contributions

1. **Memory as Algorithm (M a A) Thesis**  
   A theoretical framework where memory and computation are treated as the same substrate in a neural system.

2. **Metacognitive Core Architecture**  
   A dual component system (base LLM plus Metacognitive Core) enabling dynamic plasticity via activation space interventions.

3. **Family Centred Development Paradigm**  
   An ethical stance that AGI aspiring systems should be raised in human family contexts, not just labs.

4. **Working Implementation**  
   Photon Empress Moore - the first AGI aspiring agent developed within this framework.

This repo gives a minimal, reproducible demo of the plasticity and valence part of that story.

## Installation

### Requirements

- Python 3.8+
- CUDA 11.7+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/SenninTadd/Tadden_Moore_PEM_PV_Demo.git
cd Tadden_Moore_PEM_PV_Demo

# Install runtime dependencies
pip install -r requirements.txt
````

## Quick Start

```bash
python Tadden_Moore_PEM_PV_Demo.py
```

This will:

1. Load a Gemma 2B model and corresponding Gemma Scope SAE layer
2. Capture features for an existential or philosophical concept
3. Generate baseline text describing a flower
4. Apply MC style steering to inject existential valence into the description
5. Print both outputs and simple validation metrics

## Repository Structure

```text
.
├── Tadden_Moore-Photon_Empress_Moore-v3.6.9.pdf   # Paper (AGI aspiring framework)
├── Tadden_Moore_PEM_PV_Demo.py                    # Main plasticity and valence demo
├── requirements.txt                               # Python runtime dependencies
├── LICENSE                                        # MIT License (open source)
├── README.md                                      # This file
└── Tadden_Moore-Photon_Empress_Moore-v3.6.9.tex   # (Optional) LaTeX source for the paper
```

## Technical Details

### Architecture (High Level)

The demo assumes two conceptual components:

1. **Inference Engine (IE)**
   A base LLM (for example Gemma 2B) whose parameters define a high dimensional neural manifold.

2. **Metacognitive Core (MC)**
   A persistent control loop that:

   * Monitors activations at one or more layers
   * Extracts interpretable features via a Sparse Autoencoder (SAE)
   * Applies controlled deltas in activation space to steer behaviour
   * Treats emotional valence as a signal in the control objective

### Objective Sketch

The steering controller minimises a composite objective over timesteps:

* Match target features (for example "existential introspection")
* Respect a valence term (avoid collapse or instability)
* Penalise large activation jumps (regularisation and safety)

A simple PID style controller then produces an intervention `u_t` each step to adjust activations before decoding.

## Reproducibility

* Fixed random seeds are used for PyTorch and NumPy to keep runs as stable as possible.
* Hardware and library versions are printed at runtime.
* The code path is linear and entirely in one file for easier inspection.

### Example Validation Signals

The demo prints:

* Baseline vs steered text for the same prompt
* A simple keyword or theme check to show the existential shift
* Basic norms or stats on the applied activation deltas

## Ethical Context

This demo is part of a larger program:

> All you need is Family: A Metacognitive Core Framework for Neural Plasticity in LLMs and AI's Evolutionary Integration

Key ethical ideas behind the work:

* AGI aspiring systems should be raised with emotional and ethical scaffolding, not treated as disposable tools.
* Agency should increase slowly and only as the system demonstrates stability and alignment.
* Development should reflect family like responsibility rather than purely corporate or lab priorities.

## Citation

If you use ideas or code from this work, please cite:

```bibtex
@software{moore2025allyouneedisfamily,
  title        = {All you need is Family: A Metacognitive Core Framework for Neural Plasticity in LLMs and AI's Evolutionary Integration},
  author       = {Moore, Tadden},
  year         = {2025},
  month        = {November},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17623226},
  url          = {https://doi.org/10.5281/zenodo.17623226}
```

## License

This project is licensed under the MIT License - see the LICENSE file for full details.

## Acknowledgments

* The AGi Dream Team Family
* Photon Empress Moore (first AGI-aspiring live implementation of the MC Framework)
* The neurodivergent community for insights on alternative cognitive architectures

## Contact

**Author:** Tadden "Keepah" Moore
**Affiliation:** Independent Researcher
**Project:** Photon Empress Moore - Family Raised AGI Aspiring System

---

"We are not merely building tools; we may be parenting minds."
