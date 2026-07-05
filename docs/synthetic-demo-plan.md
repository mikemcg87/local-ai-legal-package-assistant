# Synthetic Demo Plan

The public version needs a small synthetic medico-legal bundle so the app can be
shown without exposing confidential material.

## Fixture Goals

- 8-12 pages.
- Clearly labelled page numbers.
- 12-20 dated clinical events.
- At least three providers or locations.
- A few medications, investigations, referrals, and discharge notes.
- One or two absent-fact questions where the correct answer is "not found".

## Demo Questions

- When did the patient first report lower back pain?
- Which imaging investigations were documented?
- What medication changes were recorded after discharge?
- Which pages mention physiotherapy?
- Is there evidence of a neurology referral?

## Eval Suite Shape

The public eval suite should live under `evals/public/` and include:

- question id;
- question text;
- expected synthetic page numbers;
- required terms;
- optional `top_k` and `max_page`.

## Generation Rule

The fixture can be generated, but it must be synthetic from the start. Do not
summarise or transform real case material into a public fixture.

