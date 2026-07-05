# Eval Suites

Private eval suites belong in `evals/private/`, which is ignored by git because
they can contain case names, page references, and facts from confidential
bundles.

Use a suite like this:

```json
{
  "case": "local_case_directory_name",
  "questions": [
    {
      "id": "q1",
      "question": "What was documented about the first admission?",
      "expected_pages": [12, 13],
      "required_terms": ["admission", "discharge"],
      "top_k": 8
    }
  ]
}
```

Run it locally with:

```bash
bundle-rag eval evals/private/suite.json --output reports/private/report.json
```

