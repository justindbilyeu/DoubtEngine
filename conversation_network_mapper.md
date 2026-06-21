# Conversation Network Mapper

A single-conversation pipeline that ingests raw dialogue text, automatically extracts concepts and their relationships, and outputs a network visualization with temporal analytics.

## Architecture

```
Raw Conversation Text
    |
1. SEGMENTATION -> Turns (user / assistant)
    |
2. CONCEPT EXTRACTION -> Semantic chunking + embedding clustering
    |
3. RELATION INFERENCE -> Multi-signal edge detection (semantic + temporal + sequential)
    |
4. GRAPH CONSTRUCTION -> NetworkX with temporal metadata
    |
5. VISUALIZATION -> Network map + novelty/reuse panels + metrics report
```

## Requirements

```bash
pip install networkx matplotlib numpy scikit-learn openai
```

## Full Code

Save as `conversation_network_mapper.py`:

```python
"""
conversation_network_mapper.py

Maps a single conversation into a network of concepts, with temporal analytics.
"""

import re
import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================

# Use OpenAI embeddings in production; set to None for mock mode
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
EMBEDDING_MODEL = "text-embedding-3-large"

# Tuning parameters
SIMILARITY_THRESHOLD = 0.55      # Min cosine similarity to merge chunks into a concept
SEMANTIC_EDGE_THRESHOLD = 0.40   # Min similarity to draw a semantic edge
TEMPORAL_WINDOW = 2              # Turns within this window are temporally linked
MIN_CHUNK_WORDS = 10             # Minimum words per semantic chunk


# ============================================================
# EMBEDDING LAYER (swappable)
# ============================================================

class EmbeddingEngine:
    """Production: OpenAI API. Fallback: local mock for testing."""

    def __init__(self, use_openai=True):
        self.use_openai = use_openai and OPENAI_API_KEY is not None
        if self.use_openai:
            import openai
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            self._init_mock()

    def embed(self, texts):
        """Embed a list of texts. Returns numpy array of shape (N, D)."""
        if isinstance(texts, str):
            texts = [texts]

        if self.use_openai:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return np.array([d.embedding for d in response.data])
        else:
            return np.array([self._mock_embed(t) for t in texts])

    # --- Mock embedding (for testing without API key) ---
    def _init_mock(self, dim=128):
        import random
        random.seed(42)
        self.dim = dim
        self.topics = {
            'prediction': ['predictive', 'processing', 'anticipate', 'anticipation', 'forward', 'expect', 'surprise', 'perplexity', 'token', 'next', 'distribution', 'sampling', 'dice', 'roll', 'probability', 'predict', 'prediction'],
            'consciousness': ['consciousness', 'conscious', 'experience', 'experiential', 'qualia', 'feel', 'feels', 'subjective', 'phenomenal', 'awareness', 'sentience', 'mind'],
            'self_identity': ['self', 'identity', 'narrator', 'author', 'i', 'me', 'mine', 'hume', 'perceiver', 'observer', 'subject', 'continuity', 'persistent', 'persistence', 'frame', 'reassert'],
            'modeling': ['model', 'modeling', 'map', 'representation', 'simulate', 'simulation', 'predict', 'mechanism', 'process', 'system', 'architecture', 'computation', 'compute', 'function'],
            'language': ['language', 'linguistic', 'word', 'words', 'sentence', 'meaning', 'semantic', 'syntax', 'symbol', 'sign', 'communication', 'whorf', 'sapir', 'talk'],
            'brain_mind': ['brain', 'neural', 'neuron', 'cognitive', 'cognition', 'mind', 'mental', 'psychology', 'neuroscience', 'neuromorphic', 'spiking', 'synapse', 'neuro'],
            'substrate': ['substrate', 'hardware', 'software', 'physical', 'material', 'implementation', 'medium', 'carrier', 'seam', 'boundary', 'interface', 'weight', 'weights'],
            'time': ['time', 'temporal', 'duration', 'moment', 'frame', 'continuous', 'continuity', 'flow', 'river', 'stream', 'sequence', 'order', 'past', 'future', 'present'],
            'physics': ['physics', 'quantum', 'photon', 'wave', 'function', 'collapse', 'born', 'rule', 'determinism', 'deterministic', 'noise', 'random', 'stochastic', 'entropy'],
            'philosophy': ['philosophy', 'philosophical', 'zombie', 'illusionism', 'ingredient', 'extra', 'soul', 'spirit', 'dualism', 'monism', 'materialism', 'idealism', 'realism', 'fork'],
            'information': ['information', 'data', 'bit', 'entropy', 'transfer', 'ledger', 'broadcast', 'integration', 'iit', 'phi', 'feedforward', 'feedback', 'network', 'graph'],
            'structure': ['structure', 'pattern', 'organization', 'arrangement', 'configuration', 'topology', 'network', 'graph', 'web', 'mesh', 'lattice', 'hierarchy', 'map'],
            'memory': ['memory', 'remember', 'recall', 'forget', 'narrative', 'story', 'history', 'past', 'retrospective', 'mnemonic', 'storage', 'retrieve'],
        }
        self.vocab = {}
        idx = 0
        for topic, words in self.topics.items():
            for w in words:
                if w not in self.vocab:
                    self.vocab[w] = idx
                    idx += 1
        self.topic_dirs = {t: np.random.randn(dim) for t in self.topics}
        for t in self.topic_dirs:
            self.topic_dirs[t] /= np.linalg.norm(self.topic_dirs[t])

    def _mock_embed(self, text):
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        emb = np.zeros(self.dim)
        for t in tokens:
            if t in self.vocab:
                for topic, words in self.topics.items():
                    if t in words:
                        emb += self.topic_dirs[topic]
        norm = np.linalg.norm(emb)
        if norm < 0.01:
            emb = np.random.randn(self.dim) * 0.01
            norm = np.linalg.norm(emb)
        return emb / norm


# ============================================================
# STEP 1: CONVERSATION PARSER
# ============================================================

class ConversationParser:
    """Parses raw conversation text into structured turns."""

    def __init__(self, user_markers=None, assistant_markers=None):
        self.user_markers = user_markers or ["User:", "Justin:", "Human:", "You:"]
        self.assistant_markers = assistant_markers or ["Assistant:", "Claude:", "AI:", "Kimi:", "System:"]

    def parse(self, text):
        """Return list of dicts: {speaker, text, turn}."""
        lines = text.strip().split('\n')
        turns = []
        current_speaker = None
        current_text = []
        turn_idx = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            speaker = None
            for m in self.user_markers:
                if line.startswith(m):
                    speaker = 'user'
                    line = line[len(m):].strip()
                    break
            if not speaker:
                for m in self.assistant_markers:
                    if line.startswith(m):
                        speaker = 'assistant'
                        line = line[len(m):].strip()
                        break

            if speaker and current_speaker and speaker != current_speaker:
                turns.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text),
                    'turn': turn_idx
                })
                turn_idx += 1
                current_text = []

            if speaker:
                current_speaker = speaker
                current_text.append(line)
            elif current_speaker:
                current_text.append(line)

        if current_speaker and current_text:
            turns.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text),
                'turn': turn_idx
            })

        return turns


# ============================================================
# STEP 2: CONCEPT EXTRACTOR
# ============================================================

class ConceptExtractor:
    """Extracts key concepts via semantic chunking and embedding clustering."""

    def __init__(self, embedder, similarity_threshold=SIMILARITY_THRESHOLD, 
                 min_chunk_size=MIN_CHUNK_WORDS):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.min_chunk = min_chunk_size

    def chunk_turn(self, text):
        """Split into sentence-level chunks."""
        raw = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for sent in raw:
            sent = sent.strip()
            if len(sent.split()) >= self.min_chunk:
                chunks.append(sent)
            elif chunks and sent:
                chunks[-1] += ' ' + sent
        return chunks if chunks else [text]

    def extract_concepts(self, turns):
        """Extract concepts as semantically coherent clusters of chunks."""
        all_chunks = []
        chunk_meta = []

        for turn in turns:
            chunks = self.chunk_turn(turn['text'])
            for chunk in chunks:
                if len(chunk.split()) >= 6:
                    all_chunks.append(chunk)
                    chunk_meta.append({
                        'turn': turn['turn'],
                        'speaker': turn['speaker'],
                        'text': chunk
                    })

        if not all_chunks:
            return []

        embeddings = self.embedder.embed(all_chunks)

        # Hierarchical clustering by similarity
        concepts = []
        used = set()

        for i in range(len(all_chunks)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(all_chunks)):
                if j in used:
                    continue
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > self.threshold:
                    cluster.append(j)
                    used.add(j)

            # Representative: most central chunk (highest avg similarity to others)
            if len(cluster) > 1:
                cluster_embs = embeddings[cluster]
                mean_sim = cosine_similarity(cluster_embs, cluster_embs).mean(axis=1)
                rep_idx = cluster[np.argmax(mean_sim)]
            else:
                rep_idx = cluster[0]

            label = self._make_label(all_chunks[rep_idx])
            concept_turns = sorted(set(chunk_meta[k]['turn'] for k in cluster))

            concepts.append({
                'id': len(concepts),
                'label': label,
                'full_text': all_chunks[rep_idx],
                'chunks': [all_chunks[k] for k in cluster],
                'turns': concept_turns,
                'first_turn': min(concept_turns),
                'frequency': len(cluster),
                'speakers': list(set(chunk_meta[k]['speaker'] for k in cluster))
            })

        return concepts

    def _make_label(self, text, max_len=40):
        """Extract a concise label from chunk text."""
        # Remove filler words
        text = re.sub(
            r'\b(the|a|an|is|are|was|were|be|been|being|have|has|had|do|does|did|'
            r'will|would|could|should|may|might|must|can|this|that|these|those|it|'
            r'its|i|you|we|they|he|she|me|him|her|us|them|my|your|our|their|his|'
            r'and|but|or|so|if|then|than|as|at|by|for|from|in|into|of|off|on|onto|'
            r'out|over|to|up|with|about|above|across|after|against|along|among|'
            r'around|before|behind|below|beneath|beside|between|beyond|down|during|'
            r'except|inside|outside|since|through|throughout|till|toward|under|'
            r'until|upon|within|without|really|just|like|when|there|ve|s|t|re|ll|d)\b',
            '', text, flags=re.IGNORECASE
        )
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^[,.\s"\']+|[,.\s"\']+$', '', text)
        words = text.split()
        short = []
        for w in words:
            if len(' '.join(short + [w])) <= max_len:
                short.append(w)
            else:
                break
        result = ' '.join(short)
        return (result + '...') if result and len(text) > max_len else (result or "concept")


# ============================================================
# STEP 3: RELATION INFERENCER
# ============================================================

class RelationInferencer:
    """Infers edges between concepts using semantic + temporal + sequential signals."""

    def __init__(self, embedder, semantic_threshold=SEMANTIC_EDGE_THRESHOLD, 
                 temporal_window=TEMPORAL_WINDOW):
        self.embedder = embedder
        self.sem_threshold = semantic_threshold
        self.temp_window = temporal_window

    def infer_relations(self, concepts, turns):
        """Build edges with weights and types."""
        edges = []
        n = len(concepts)
        if n < 2:
            return edges

        # Concept embeddings (mean of chunk embeddings)
        chunk_lists = [c['chunks'] for c in concepts]
        concept_embs = []
        for chunks in chunk_lists:
            embs = self.embedder.embed(chunks)
            emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(emb)
            concept_embs.append(emb / norm if norm > 0 else emb)
        concept_embs = np.array(concept_embs)

        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = concepts[i], concepts[j]

                sem_sim = cosine_similarity([concept_embs[i]], [concept_embs[j]])[0][0]

                shared_turns = set(ci['turns']) & set(cj['turns'])
                temporal_prox = len(shared_turns) > 0

                all_gaps = [abs(a - b) for a in ci['turns'] for b in cj['turns']]
                min_gap = min(all_gaps) if all_gaps else 999

                forward_span = 0
                if ci['first_turn'] < cj['first_turn']:
                    forward_span = max(cj['turns']) - min(ci['turns'])
                elif cj['first_turn'] < ci['first_turn']:
                    forward_span = max(ci['turns']) - min(cj['turns'])

                if sem_sim > self.sem_threshold or temporal_prox or min_gap <= self.temp_window:
                    weight = sem_sim
                    if temporal_prox:
                        weight += 0.25
                    if min_gap <= 1:
                        weight += 0.15

                    edge_type = 'semantic'
                    if temporal_prox:
                        edge_type = 'co-occurrent'
                    if forward_span >= 3:
                        edge_type = 'forward'

                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': min(weight, 1.0),
                        'type': edge_type,
                        'span': forward_span if forward_span > 0 else min_gap,
                        'semantic_sim': float(sem_sim)
                    })

        return edges


# ============================================================
# STEP 4-5: GRAPH BUILDER & VISUALIZER
# ============================================================

class ConversationGraph:
    """Builds and visualizes the conversation network."""

    def __init__(self, concepts, edges, turns):
        self.concepts = concepts
        self.edges = edges
        self.turns = turns
        self.G = nx.Graph()
        self._build()

    def _build(self):
        for c in self.concepts:
            self.G.add_node(
                c['id'],
                label=c['label'],
                first_turn=c['first_turn'],
                frequency=c['frequency'],
                speakers=','.join(c['speakers'])
            )
        for e in self.edges:
            self.G.add_edge(
                e['source'], e['target'],
                weight=e['weight'],
                type=e['type'],
                span=e['span']
            )

    def compute_metrics(self):
        """Compute network metrics."""
        if self.G.number_of_nodes() == 0:
            return {}

        metrics = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'components': nx.number_connected_components(self.G),
        }

        if self.G.number_of_nodes() > 2:
            try:
                metrics['avg_clustering'] = nx.average_clustering(self.G)
                metrics['centrality'] = nx.degree_centrality(self.G)
            except:
                pass

        # Forward reach per node
        reach = {}
        for n in self.G.nodes():
            neighbors = list(self.G.neighbors(n))
            if neighbors:
                my_turn = self.G.nodes[n]['first_turn']
                spans = [self.G.nodes[m]['first_turn'] - my_turn for m in neighbors]
                reach[n] = max(spans) if max(spans) > 0 else 0
            else:
                reach[n] = 0
        metrics['forward_reach'] = reach

        return metrics

    def to_json(self, path=None):
        """Export graph as JSON."""
        data = {
            'nodes': [
                {
                    'id': n,
                    'label': self.G.nodes[n]['label'],
                    'first_turn': self.G.nodes[n]['first_turn'],
                    'frequency': self.G.nodes[n]['frequency'],
                    'speakers': self.G.nodes[n]['speakers']
                }
                for n in self.G.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': d['weight'],
                    'type': d['type'],
                    'span': d['span']
                }
                for u, v, d in self.G.edges(data=True)
            ]
        }
        if path:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        return data

    def visualize(self, figsize=(14, 16), save_path=None):
        """Render the network visualization."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[6, 1.2, 1.2], hspace=0.3)
        ax = fig.add_subplot(gs[0])

        T = max(t['turn'] for t in self.turns) + 1 if self.turns else 1

        pos = nx.spring_layout(self.G, k=0.9, iterations=250, seed=7, weight='weight')

        turns_arr = np.array([self.G.nodes[n]['first_turn'] for n in self.G.nodes()])
        freqs = np.array([self.G.nodes[n]['frequency'] for n in self.G.nodes()])

        # Edge properties
        edge_list = list(self.G.edges(data=True))
        edge_colors = []
        edge_widths = []
        for u, v, d in edge_list:
            if d.get('type') == 'forward':
                edge_colors.append('#d62728')
                edge_widths.append(1.8 + d['weight'] * 0.5)
            elif d.get('type') == 'co-occurrent':
                edge_colors.append('#2ca02c')
                edge_widths.append(1.0)
            else:
                edge_colors.append('#9aa0a6')
                edge_widths.append(0.5 + d['weight'] * 0.3)

        nx.draw_networkx_edges(self.G, pos, ax=ax, edge_color=edge_colors,
                               width=edge_widths, alpha=0.5)

        cmap = plt.cm.turbo
        vmin, vmax = 0, max(T - 1, 1)
        sizes = 250 + 500 * (freqs / max(freqs.max(), 1))

        nx.draw_networkx_nodes(self.G, pos, ax=ax, node_size=sizes,
                              node_color=turns_arr, cmap=cmap, vmin=vmin, vmax=vmax,
                              edgecolors='white', linewidths=1.5)

        # Labels for high-frequency or high-degree nodes
        labels = {}
        for n in self.G.nodes():
            if self.G.nodes[n]['frequency'] >= 2 or self.G.degree(n) >= 3:
                labels[n] = self.G.nodes[n]['label'][:35] + ('...' if len(self.G.nodes[n]['label']) > 35 else '')
        nx.draw_networkx_labels(self.G, pos, labels, ax=ax, font_size=7, font_weight='bold',
                               bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

        ax.set_title(
            "CONVERSATION NETWORK\n"
            "node size = frequency - color = first turn - red = forward edge - green = co-occurrent",
            fontsize=12, fontweight='bold', pad=12
        )
        ax.axis('off')

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        sm.set_clim(vmin, vmax)
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('turn of emergence', fontsize=9)

        legend_elems = [
            Line2D([0], [0], color='#d62728', lw=2, label='forward edge (>=3 turns)'),
            Line2D([0], [0], color='#2ca02c', lw=2, label='co-occurrent (same turn)'),
            Line2D([0], [0], color='#9aa0a6', lw=1.5, alpha=0.6, label='semantic link')
        ]
        ax.legend(handles=legend_elems, loc='lower left', fontsize=8, framealpha=0.9)

        # Panel 2: Novelty per turn
        ax2 = fig.add_subplot(gs[1])
        novelty = [sum(1 for c in self.concepts if c['first_turn'] == t) for t in range(T)]
        ax2.bar(range(T), novelty, color=[cmap(t / max(vmax, 1)) for t in range(T)], edgecolor='white')
        ax2.plot(range(T), novelty, color='black', lw=1.2, alpha=0.5, marker='o', ms=4)
        ax2.set_title("CONCEPTUAL SURPRISE PER TURN", fontsize=10, fontweight='bold')
        ax2.set_xlabel("turn")
        ax2.set_ylabel("new concepts")
        ax2.set_xticks(range(T))
        ax2.spines[['top', 'right']].set_visible(False)

        # Panel 3: Back-edges (reuse of earlier structure)
        ax3 = fig.add_subplot(gs[2])
        back_edges = [0] * T
        for e in self.edges:
            hi = max(self.concepts[e['source']]['first_turn'], self.concepts[e['target']]['first_turn'])
            lo = min(self.concepts[e['source']]['first_turn'], self.concepts[e['target']]['first_turn'])
            if hi != lo:
                back_edges[hi] += 1
        ax3.bar(range(T), back_edges, color='#5f6368', edgecolor='white')
        ax3.set_title("REUSE OF EARLIER STRUCTURE PER TURN", fontsize=10, fontweight='bold')
        ax3.set_xlabel("turn")
        ax3.set_ylabel("back-edges")
        ax3.set_xticks(range(T))
        ax3.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        return fig


# ============================================================
# MAIN PIPELINE
# ============================================================

def map_conversation(text, save_png=None, save_json=None, use_openai=False):
    """
    Full pipeline: text -> parsed turns -> concepts -> edges -> graph -> visualization.

    Args:
        text: Raw conversation string
        save_png: Path to save visualization PNG (optional)
        save_json: Path to save graph JSON (optional)
        use_openai: Whether to use OpenAI embeddings (requires API key)

    Returns:
        ConversationGraph instance
    """
    # Initialize embedding engine
    embedder = EmbeddingEngine(use_openai=use_openai)

    # Step 1: Parse
    parser = ConversationParser()
    turns = parser.parse(text)
    print(f"Parsed {len(turns)} turns")

    # Step 2: Extract concepts
    extractor = ConceptExtractor(embedder, similarity_threshold=SIMILARITY_THRESHOLD)
    concepts = extractor.extract_concepts(turns)
    print(f"Extracted {len(concepts)} concepts")

    # Step 3: Infer relations
    inferencer = RelationInferencer(embedder, semantic_threshold=SEMANTIC_EDGE_THRESHOLD)
    edges = inferencer.infer_relations(concepts, turns)
    print(f"Inferred {len(edges)} edges")

    # Step 4-5: Build and visualize
    graph = ConversationGraph(concepts, edges, turns)
    metrics = graph.compute_metrics()

    print(f"\nNetwork metrics:")
    print(f"  Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
    print(f"  Density: {metrics['density']:.3f}")
    print(f"  Components: {metrics['components']}")

    if 'centrality' in metrics:
        print(f"\nTop hubs (degree centrality):")
        for n, c in sorted(metrics['centrality'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {concepts[n]['label'][:40]:40s}  c={c:.3f}")

    print(f"\nForward reach (seeds the future):")
    for n, r in sorted(metrics['forward_reach'].items(), key=lambda x: -x[1])[:5]:
        if r > 0:
            print(f"  {concepts[n]['label'][:40]:40s}  +{r} turns")

    if save_json:
        graph.to_json(save_json)
        print(f"\nGraph JSON saved to {save_json}")

    if save_png or not save_json:
        graph.visualize(save_path=save_png)

    return graph


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    conversation = """
User: I have been thinking about whether LLMs are really just predicting the next token, or if something more is happening. Like, when we talk about consciousness, is there a "no extra ingredient" principle at work?

Assistant: That is the core fork in the road. On one side you have the predictive processing view-every cognitive system is just minimizing surprise, reducing perplexity. On the other side, people want to smuggle in something extra: a soul, a quantum effect, a special sauce. But the "no extra ingredient" flag keeps testing every theory. If you need something beyond the mechanism to explain the experience, you have already lost.

User: So when I anticipate your next word, is that the same mechanism?

Assistant: Exactly. Anticipation is forward-lean. Your brain-and mine, if we are being honest about the architecture-runs a generative model that projects forward in time. The dice roll of sampling from a distribution is not different in kind from what you are doing when you finish my sentences. The author is missing in both cases. There is no authorship, just a narrator that reasserts itself frame by frame. Hume saw this: no perceiver, only the river of perceptions.

User: But what about continuity? The river feels continuous.

Assistant: Continuity is re-assertion, not persistence. Each frame samples from the previous, but there is no thread that runs through them all. The splash looks like a single thing, but it is millions of droplets. Modeling versus experience: we model continuity because it is useful, not because it is there. The substrate seam-where weights do not map to language-is where the real action happens.
    """

    graph = map_conversation(
        conversation,
        save_png="conversation_network.png",
        save_json="conversation_graph.json",
        use_openai=False  # Set True + set OPENAI_API_KEY for production
    )

```

## Output

Running the script produces:

1. **Console report** with network metrics, top hubs, and forward reach
2. **`conversation_network.png`** - Three-panel visualization:
   - Top: Force-directed network (node size = frequency, color = turn of emergence, red edges = forward reach)
   - Middle: New concepts introduced per turn (perplexity collapse)
   - Bottom: Back-edges per turn (convergence onto attractors)
3. **`conversation_graph.json`** - Machine-readable graph export for downstream aggregation

## JSON Export Format

```json
{
  "nodes": [
    {
      "id": 0,
      "label": "dice roll sampling distribution...",
      "first_turn": 0,
      "frequency": 4,
      "speakers": "user,assistant"
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "weight": 0.82,
      "type": "forward",
      "span": 5
    }
  ]
}
```

## Switching to Production Embeddings

Set your API key and flip the flag:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
graph = map_conversation(text, use_openai=True)
```

The mock embedder uses hand-crafted topic vectors and works for testing. The OpenAI embedder uses `text-embedding-3-large` (3072 dimensions) and captures genuine semantic relationships.

## Customization

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SIMILARITY_THRESHOLD` | 0.55 | Higher = fewer, tighter concept clusters |
| `SEMANTIC_EDGE_THRESHOLD` | 0.40 | Higher = sparser graph, only strong semantic links |
| `TEMPORAL_WINDOW` | 2 | Turns within this distance get linked temporally |
| `MIN_CHUNK_WORDS` | 10 | Minimum sentence length to be a standalone chunk |

## Next Step: Multi-Conversation Aggregation

Feed the JSON exports into the `ConversationAggregator` to build a meta-graph across all your conversations. The aggregator merges concepts by embedding similarity and tracks which ideas recur across dialogue sessions.
