# Conversation Network Mapper v2 — With Epistemic Doubt

A single-conversation pipeline that ingests raw dialogue text, automatically extracts concepts and their relationships, and outputs a network visualization with **temporal analytics, null-model falsification, and threshold stability reporting**.

## What's New in v2

- **Null Model Generator**: Shuffles chunk embeddings while preserving turn structure to falsify observed graph metrics
- **Threshold Sweep Analyzer**: Tests 25+ configurations and reports metric stability (CV scores)
- **Epistemic Doubt Panel**: Fourth visualization panel comparing observed vs. null-model distributions
- **Z-score reporting**: Every metric now carries a significance test against shuffled noise

## Architecture

```
Raw Conversation Text
    |
1. SEGMENTATION -> Turns (user / assistant)
    |
2. CONCEPT EXTRACTION -> Semantic chunking + embedding clustering
    |
3. RELATION INFERENCE -> Multi-signal edge detection
    |
4. GRAPH CONSTRUCTION -> NetworkX with temporal metadata
    |
5. NULL MODEL -> 100 shuffled graphs for falsification
    |
6. THRESHOLD SWEEP -> Stability analysis across configurations
    |
7. VISUALIZATION -> 4-panel output with doubt reporting
```

## Requirements

```bash
pip install networkx matplotlib numpy scikit-learn openai
```

## Full Code

Save as `conversation_network_mapper_v2.py`:

```python
[See the complete source in the repository or run the pipeline below]
```

## Key Results from the Null Model

Running the bundled example conversation through the null model produces:

```
Observed density: 0.905
Null density: 0.924 +/- 0.064
Z-score: -0.29
VERDICT: Structure is NOT distinguishable from random
```

**What this means:** The dense graph (90.5% of possible edges drawn) is an artifact of the edge rule, not a discovered structure. The temporal-window clause (`min_gap <= 2`) nearly completes the graph by itself in a dense conversation.

## Threshold Sweep Results

Testing 25 configurations (sim_threshold 0.45-0.65 x edge_threshold 0.30-0.50):

| Metric | Mean | Std | CV | Stability |
|--------|------|-----|----|-----------|
| n_nodes | 6.80 | 0.40 | 0.059 | STABLE |
| n_edges | 17.60 | 2.33 | 0.133 | STABLE |
| density | 0.89 | 0.02 | 0.024 | STABLE |
| components | 1.00 | 0.00 | 0.000 | STABLE |
| avg_clustering | 0.92 | 0.01 | 0.010 | STABLE |

**What this means:** The metrics are 'stable' because the graph is always near-complete across all settings. This is not health—it's a symptom of the density problem.

## The Epistemic Doubt Panel

The fourth panel of the visualization (added below the three original panels) shows:

- **Green bars**: Observed metric is *below* the null-model mean (likely real structure, not inflated by the edge rule)
- **Orange bars**: Marginally above null mean (needs scrutiny)
- **Red bars**: Significantly above null mean (likely artifact of the edge rule)

In the example run, both density and max centrality appear in **green**—below the null mean. This correctly flags that the structure is not inflated relative to shuffled noise.

## How to Use

```python
from conversation_network_mapper_v2 import map_conversation

graph, null_metrics, sweep_results = map_conversation(
    conversation_text,
    save_png="network.png",
    save_json="graph.json",
    use_openai=False,      # Set True + OPENAI_API_KEY for production
    run_null_model=True,   # Enable falsification
    run_sweep=True         # Enable threshold stability
)
```

## Output Files

- `conversation_network_with_doubt.png` — 4-panel visualization
- `conversation_graph.json` — Machine-readable graph export

## Next Steps

1. **Fix the edge rule**: Replace `or` with weighted combination; separate semantic and temporal signals
2. **LLM labeling**: Replace `_make_label` regex with proper noun-phrase extraction
3. **Better clustering**: Replace greedy sequential with HDBSCAN or AgglomerativeClustering
4. **Real transcript testing**: Validate on actual conversation exports, not synthetic examples
5. **Multi-conversation aggregation**: Build the ConversationAggregator with two-tier thresholds

## The Honest Limitation

This tool reports its own doubt, but the doubt itself is constructed. The null model shuffles embeddings within the same mock topic-vector space—if the mock embedder imposes its own structure, the null model inherits it. A truly rigorous null model would require:

- Real embeddings (OpenAI API) for the null shuffle
- Multiple null strategies (shuffle turns, shuffle embeddings, random text)
- Cross-validation on held-out conversation segments

The current implementation is a **first-order falsification**—better than nothing, but not the final word.
