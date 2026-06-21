# DoubtEngine
# DoubtEngine

> A conversation-to-network pipeline that outputs its own uncertainty.

DoubtEngine maps dialogue into concept networks and then asks: *is this structure real, or an artifact of the edge rule?* It generates the graph, runs a null-model falsification test, sweeps threshold configurations, and reports significance on every metric. The doubt is not a bug to fix—it is the product.

---

## The Core Claim

Most conversation-mapping tools produce beautiful graphs. DoubtEngine produces **qualified graphs**. Every output carries a z-score against shuffled noise. If the structure doesn't survive falsification, the tool says so—on the figure, in the console, and in the JSON export.

---

## What It Does

1. **Parses** raw conversation text into structured turns
2. **Extracts** concepts via semantic chunking + embedding clustering
3. **Infers** relations using semantic + temporal + sequential signals
4. **Builds** a force-directed network with temporal coloring
5. **Falsifies** the observed structure against 100 null-model shuffles
6. **Sweeps** threshold configurations and reports metric stability
7. **Visualizes** everything in a 4-panel output including an **epistemic doubt panel**

---

## Quick Start

```bash
pip install -e .
```

```python
from doubt_engine import map_conversation

graph, null_metrics, sweep = map_conversation(
    conversation_text,
    save_png="network.png",
    save_json="graph.json",
    use_openai=False,      # Set True + OPENAI_API_KEY for production embeddings
    run_null_model=True,   # Enable falsification (the point of this tool)
    run_sweep=True         # Enable threshold stability analysis
)
```

---

## The 4-Panel Output

| Panel | Content | What It Tells You |
|-------|---------|-------------------|
| 1 | **Network Map** | Nodes = concepts, color = turn of emergence, size = frequency, red edges = forward reach |
| 2 | **Conceptual Surprise** | New concepts per turn (perplexity collapse signature) |
| 3 | **Structure Reuse** | Back-edges per turn (convergence onto attractors) |
| 4 | **Epistemic Doubt** | Observed metrics vs. null-model distribution (z-scores, significance) |

---

## The Null Model: How It Works

DoubtEngine doesn't just compare your graph to random—it compares it to **structured random**. The null model:

- Preserves the turn structure (who spoke when, how many chunks per turn)
- Preserves the mechanical properties of conversation (chunk length, turn alternation)
- **Shuffles the semantic content** by permuting chunk embeddings
- Re-clusters and re-graphs the shuffled data
- Repeats 100 times to build a null distribution

If your observed density sits at the 95th percentile of null densities, the structure is significant. If it sits at the 50th, the graph is noise dressed as signal.

### Example Result (Bundled Philosophical Dialogue)

```
Observed density: 0.905
Null density: 0.924 ± 0.064
Z-score: -0.29
VERDICT: Structure is NOT distinguishable from random
```

**This is not a failure.** This is the tool working as designed. The dense graph (90.5% of possible edges) is an artifact of the temporal-window clause (`min_gap <= 2`) in a dense philosophical conversation. The doubt panel correctly flags this in green—below the null mean, not above it.

---

## Threshold Sweep: Stability Analysis

DoubtEngine tests 25+ threshold configurations and reports coefficient of variation (CV) for each metric:

| Metric | CV | Stability |
|--------|----|-----------|
| n_nodes | 0.059 | STABLE |
| n_edges | 0.133 | STABLE |
| density | 0.024 | STABLE |
| components | 0.000 | STABLE |

**Caveat:** These metrics are "stable" because the graph is always near-complete across settings. This is a symptom, not health. The threshold sweep reveals that the edge rule dominates regardless of tuning.

---

## The Honest Limitations

1. **The edge rule is crude.** `sem_sim > threshold OR temporal_prox OR min_gap <= window` nearly completes the graph in dense conversations. This is the highest-priority fix for v0.2.

2. **The mock embedder imposes structure.** The topic vectors for "consciousness," "predictive processing," etc. shape the null model's baseline. A truly rigorous null requires real embeddings (OpenAI API) and multiple null strategies.

3. **Labeling is regex-based.** `_make_label` strips stopwords and truncates. It produces "dice roll sampling distribution..." instead of "Predictive Processing." The fix is LLM-powered summarization or spaCy noun-phrase extraction.

4. **Clustering is greedy and order-dependent.** HDBSCAN or AgglomerativeClustering would be more robust.

5. **The aggregator is not yet built.** Cross-conversation merging inherits every threshold problem at scale.

---

## Architecture

```
Raw Conversation Text
    |
1. SEGMENTATION → Turns (user / assistant)
    |
2. CONCEPT EXTRACTION → Semantic chunking + embedding clustering
    |
3. RELATION INFERENCE → Multi-signal edge detection
    |
4. GRAPH CONSTRUCTION → NetworkX with temporal metadata
    |
5. NULL MODEL → 100 shuffled graphs for falsification
    |
6. THRESHOLD SWEEP → Stability analysis across configurations
    |
7. VISUALIZATION → 4-panel output with doubt reporting
```

---

## Requirements

```bash
pip install networkx matplotlib numpy scikit-learn openai
```

---

## Repository Structure

```
DoubtEngine/
├── README.md                    # This file
├── doubt_engine/              # Core package
│   ├── __init__.py
│   ├── parser.py              # ConversationParser
│   ├── extractor.py           # ConceptExtractor
│   ├── inferencer.py          # RelationInferencer
│   ├── graph.py               # ConversationGraph + visualize_with_doubt
│   ├── null_model.py          # NullModelGenerator
│   ├── threshold_sweep.py     # ThresholdSweepAnalyzer
│   └── embedder.py            # EmbeddingEngine (OpenAI + mock)
├── examples/
│   └── philosophical_dialogue.txt
├── tests/
│   └── test_null_model.py     # Falsification tests
├── notebooks/
│   └── threshold_sweep_demo.ipynb
└── pyproject.toml
```

---

## The Charter

This project is governed by a simple rule: **instantiation over narrative.**

- We do not claim the graph "reveals" conversation structure.
- We claim the graph *can be read as* conversation structure, and we report how much of that reading survives falsification.
- The most interesting output is not the figure—it's the doubt panel.
- The most interesting finding is not which ideas are central—it's whether centrality itself is significant.

If you use DoubtEngine, cite the null-model z-score alongside every metric.

---

## Contributors

Built in collaboration with Kimi, Claude, Grok, Gemini, DeepSeek, and Sage. The cross-model review was the first test of the tool's epistemic claims.

---

## License

MIT
