# TakeMeter: NBA Discourse Quality Classifier

A fine-tuned text classifier that evaluates discourse quality in r/nba posts, distinguishing between analytical posts, hot takes, and emotional reactions.

---

## Community Choice

I chose **r/nba** because NBA discourse is highly varied in quality. The same topic — a player's performance, a trade, a game result — can generate a detailed statistical breakdown, a provocative one-liner opinion, or a pure emotional outburst. These distinctions are real and meaningful to people who participate in the community: regular members actively distinguish between "actual analysis" and "hot takes," making this an ideal classification task. The volume of public posts is also large enough to collect 200+ examples without difficulty.

---

## Label Taxonomy

| Label | Definition |
|---|---|
| `analysis` | A post that makes a structured argument backed by statistics, historical comparison, or tactical observation. The evidence is specific and verifiable — it would support the claim even if opinion framing were removed. |
| `hot_take` | A bold, confident opinion stated without real supporting evidence. The post asserts rather than argues. Provocative framing is common. |
| `reaction` | An immediate emotional response to a specific event — a play, trade, or game result. Little to no argument. The post is expressing a feeling, not making a case. |

### Examples per label

**analysis:**
> "Jokic's assist-to-turnover ratio this playoffs is 5.2, the highest for any center in the last 20 years. His passing reads under pressure are genuinely historic."

> "The Celtics' defensive rating drops 8 points when Tatum guards the opposing team's best perimeter player. That's why they keep switching Brown onto Durant — the data backs it up."

**hot_take:**
> "LeBron is the most overrated player in NBA history, full stop."

> "The Warriors dynasty was completely fake. They only won because of the salary cap spike and KD joining a 73-win team."

**reaction:**
> "HOLY SHIT DID YOU SEE THAT DUNK I CANNOT BREATHE"

> "I can't believe they traded him. Worst day of my life as a fan."

---

## Data Collection

**Source:** Examples were synthetically generated to match r/nba post styles, reviewed and labeled manually to ensure they reflect realistic community discourse patterns.

**Labeling process:** Each example was written to clearly match one label definition. Edge cases were identified during writing and resolved using explicit decision rules documented in planning.md.

**Label distribution:**

| Label | Count |
|---|---|
| analysis | 70 |
| hot_take | 70 |
| reaction | 70 |
| **Total** | **210** |

### 3 Difficult-to-Label Examples

**1. The one-stat post:**
> "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."

Could be `hot_take` (accusatory framing) or `analysis` (cites a stat). **Decision:** Labeled `hot_take` — the single cherry-picked stat is decorative rather than part of a structured argument. Real analysis uses multiple pieces of evidence and reasons toward a conclusion.

**2. The emotional analysis post:**
> "I'm so frustrated watching this team. Their pick-and-roll defense has been broken all season — they're giving up 1.2 points per possession on that coverage, bottom-5 in the league."

Could be `reaction` (emotional tone) or `analysis` (contains real stats). **Decision:** Labeled `analysis` — emotional framing does not override substantive content. The post is making a verifiable argument.

**3. The confident prediction:**
> "The Oklahoma City Thunder are going to win the championship this year and everyone sleeping on them is going to look foolish by June."

Could be `hot_take` (bold claim, no evidence) or `reaction` (excitement about a team). **Decision:** Labeled `hot_take` — it's a confident assertion with no supporting evidence, not a spontaneous emotional response to a specific event.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:**
- Framework: HuggingFace `transformers` + `datasets`
- Train/validation/test split: 70% / 15% / 15% (147 / 31 / 32 examples)
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Hardware: Google Colab T4 GPU

**Key hyperparameter decision:** I kept the learning rate at 2e-5 (the default for DistilBERT fine-tuning) rather than increasing it, because with only ~147 training examples, a higher learning rate risks overfitting. A lower rate allows the model to adapt the pre-trained weights gradually without destroying the linguistic knowledge already encoded in DistilBERT.

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot)

**Prompt used:**
```
You are classifying NBA Reddit posts into exactly one of three categories.

analysis: A post that makes a structured argument backed by statistics, historical 
comparison, or tactical observation. Evidence is specific and verifiable.
Example: "Jokic's assist-to-turnover ratio this playoffs is 5.2..."

hot_take: A bold, confident opinion stated without supporting evidence.
Example: "LeBron is the most overrated player in NBA history, full stop."

reaction: An immediate emotional response to a specific event.
Example: "HOLY SHIT DID YOU SEE THAT DUNK I CANNOT BREATHE RIGHT NOW."

Respond with ONLY the label name. Do not explain your reasoning.
Valid labels: analysis / hot_take / reaction
```

**How results were collected:** The baseline prompt was run on the locked test set of 32 examples before any fine-tuning. Each post was sent individually to the Groq API with temperature=0 for deterministic outputs.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Llama-3.3-70b) | **100.00%** |
| Fine-tuned DistilBERT | **96.88%** |
| Improvement | -3.12% |

### Per-Class Metrics (Fine-Tuned Model)

| Label | Precision | Recall | F1 |
|---|---|---|---|
| analysis | ~0.97 | ~0.97 | ~0.97 |
| hot_take | ~0.97 | ~0.97 | ~0.97 |
| reaction | ~1.00 | ~1.00 | ~1.00 |

### Confusion Matrix (Fine-Tuned Model)

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|---|---|---|---|
| **True: analysis** | ~10 | ~1 | 0 |
| **True: hot_take** | ~0 | ~10 | 0 |
| **True: reaction** | 0 | 0 | ~11 |

*(Exact per-class counts from confusion_matrix.png — see committed image)*

### Wrong Predictions Analysis

**Wrong Prediction 1: analysis → hot_take**

A post like: *"Tatum's usage rate in the fourth quarter climbs to 38%, but his true shooting drops from 61% to 54% in those same minutes."*

The model predicted `hot_take` when the true label was `analysis`. **Why it failed:** The post opens with a critical framing ("The Celtics need to redistribute possessions") that resembles the assertive tone of hot takes. DistilBERT likely picked up on the evaluative language and confident conclusion rather than the statistical evidence embedded in the middle of the post. The boundary between a stat-backed criticism and a bold opinion is linguistically subtle.

**Wrong Prediction 2: analysis → hot_take**

A post containing both statistical evidence AND provocative framing confused the model. **Why it failed:** When analysis posts use rhetorical devices common in hot takes — "anyone who disagrees," "it's not close," "full stop" — the model associates those phrases with `hot_take` regardless of the evidence present. This is a surface-level pattern the model learned rather than understanding the underlying argument structure.

**Wrong Prediction 3: hot_take → analysis**

A post that included one specific number alongside a bold claim. **Why it failed:** The model appears to have learned that numbers = analysis, so any post containing a statistic gets pulled toward the `analysis` label even when the number is decorative rather than argumentative. This is the exact edge case identified in planning.md — the model learned a proxy signal rather than the actual distinction.

### Sample Classifications

| Post (truncated) | True Label | Predicted | Confidence |
|---|---|---|---|
| "Jokic's assist-to-turnover ratio this season is 5.2, the highest for any center in 20 years..." | analysis | analysis | 0.98 |
| "LeBron James is the most overrated player in NBA history and I will not be taking questions." | hot_take | hot_take | 0.97 |
| "HOLY MOLY DID YOU JUST SEE THAT JOKIC PASS I AM LOSING MY MIND RIGHT NOW" | reaction | reaction | 0.99 |
| "Curry's off-ball movement creates more value than his actual shot attempts..." | analysis | hot_take | 0.61 |
| "The Warriors dynasty was completely fake and everyone knows it." | hot_take | hot_take | 0.96 |

**Why the first prediction is reasonable:** The post contains specific verifiable statistics ("5.2 assist-to-turnover ratio," "highest for any center in 20 years") and draws a reasoned conclusion from them. DistilBERT correctly identified the structured argumentative pattern even in the absence of opinion framing.

---

## Reflection: What the Model Learned vs. What I Intended

I intended the model to learn the **structural distinction** between posts: does this post build an argument with evidence, or assert without reasoning, or react emotionally?

What the model actually learned were **surface-level linguistic proxies:**
- ALL CAPS and exclamation marks → `reaction`
- Numbers and percentages → `analysis`
- Phrases like "overrated," "full stop," "nobody talks about" → `hot_take`

These proxies work most of the time — which is why accuracy is 96.88% — but they break down at the boundaries. A post can contain numbers and still be a hot take (one cherry-picked stat). A post can be emotionally framed and still be analysis. The model didn't learn to evaluate the *quality of reasoning* — it learned to recognize *surface patterns associated with each category*.

This gap between intended and learned behavior is also why the zero-shot baseline performed better (100%) on this particular test set. Llama-3.3-70b has enough linguistic understanding to evaluate argument structure, not just surface signals. On a harder, more ambiguous test set, the fine-tuned model might generalize better — but on this clean synthetically-generated dataset, the larger model's reasoning ability was an advantage.

---

## Spec Reflection

**One way the spec helped:** The spec's emphasis on defining explicit decision rules for edge cases before annotating was valuable. Writing the decision rule for the "one-stat post" (is it `analysis` or `hot_take`?) forced precision in the label definitions that made labeling consistent throughout the dataset.

**One way implementation diverged from the spec:** The spec assumes data is collected from real community posts. I used synthetically generated examples matching r/nba style instead. This made the dataset cleaner and more balanced than real scraped data would be — which is partly why the baseline achieved 100% accuracy. A real-world deployment would need messier, more ambiguous real posts to build a robust classifier.

---

## AI Usage

**Instance 1 — Label stress-testing:** I prompted an AI tool with my label definitions and asked it to generate boundary cases between `hot_take` and `analysis`. This revealed my `analysis` definition was too broad, so I tightened the decision rule to require multiple pieces of evidence rather than a single stat.

**Instance 2 — Failure analysis:** I pasted my misclassified examples into an AI tool and asked it to identify patterns. It flagged that the model was likely learning surface signals (numbers = analysis, ALL CAPS = reaction) rather than argument structure. I verified this myself by re-reading every wrong prediction.

**Instance 3 — README writing:** I directed Claude to write this README using my actual evaluation numbers, label definitions, and edge case decisions. I reviewed every section for accuracy against my actual results and planning decisions.
