# Eval Plan

## Why Evals Come First

The product failed to become reliably useful for client work because it was hard to tell whether changes actually improved the expert witness workflow. Better models may help, but they need to be measured against the actual tasks:

- finding the right page;
- retrieving the relevant source text;
- preserving factual accuracy;
- not hallucinating absent facts;
- producing useful chronology entries.

## Eval Types

### Retrieval Eval

Measures whether the retrieval stack returns the expected source pages and source terms for a question.

Inputs:

- case name;
- question;
- expected page numbers;
- required terms;
- optional max page;
- top-k;
- semantic/BM25 weights.

Outputs:

- top retrieved pages;
- expected page hit;
- mean reciprocal rank for expected pages;
- required-term coverage;
- retrieved chunks and scores.

### Answer Eval

Later milestone. Measures whether the local model answer is faithful to retrieved source chunks.

Checks:

- cites pages that were retrieved;
- includes required facts;
- does not introduce unsupported facts;
- says "not found" when retrieval is insufficient.

### Timeline Eval

Measures chronology extraction quality.

Checks:

- expected dates found;
- event type correct;
- provider/location extracted;
- duplicate events merged correctly;
- false positives suppressed.

## Private Suite Format

Private suites live in `evals/private/` and must not be committed.

```json
{
  "case": "example_case_name",
  "questions": [
    {
      "id": "initial-presentation",
      "question": "When did the patient first present with chest pain?",
      "expected_pages": [12, 13],
      "required_terms": ["chest pain", "presented"],
      "top_k": 10
    }
  ]
}
```

## Public Suite Format

Public suites should use synthetic fixtures only. They can live in `evals/public/` once a synthetic case fixture exists.

## Metrics

### Page Hit

`true` if any retrieved chunk page appears in `expected_pages`.

### Page MRR

Mean reciprocal rank for the first retrieved expected page.

Example:

- expected page first appears at rank 1: `1.0`
- rank 2: `0.5`
- rank 5: `0.2`
- not found: `0.0`

### Required Term Coverage

Percentage of required terms found across retrieved chunk text.

## Working Rule

Before changing chunking, retrieval, reranking, or model defaults:

1. add or update eval questions;
2. run the baseline;
3. make the change;
4. compare results;
5. keep the change only if it improves the target workflow or has a clear trade-off.
