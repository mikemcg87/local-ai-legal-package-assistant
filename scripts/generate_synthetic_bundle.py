"""Generate the synthetic medico-legal demo bundle.

Every person, institution, date, and clinical fact below is invented from
scratch for the public demo fixture. Nothing is derived from real case
material (see docs/synthetic-demo-plan.md).

Uses reportlab: fpdf2 output made Docling report doubled page numbers
(1, 3, 5, ...), which broke page attribution; reportlab pages number 1..N.

Usage:
    python scripts/generate_synthetic_bundle.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "evals" / "public" / "fixtures" / "synthetic-bundle.pdf"

TOTAL_PAGES = 10
PAGE_W, PAGE_H = A4
MARGIN = 56
BODY_WIDTH = PAGE_W - 2 * MARGIN

PAGES: list[tuple[str, str]] = [
    (
        "BUNDLE INDEX",
        """In the matter of: Margaret Doyle (Claimant)

Prepared for demonstration purposes. All contents synthetic.

Index of documents:

Page 2 - GP records, Rathmore Medical Practice (extract 1)
Page 3 - GP records, Rathmore Medical Practice (extract 2)
Page 4 - Referral letter to Orthopaedic Outpatients
Page 5 - Physiotherapy initial assessment
Page 6 - Physiotherapy progress note
Page 7 - MRI lumbar spine report
Page 8 - Hospital admission and operation note
Page 9 - Discharge summary
Page 10 - Witness statement of the claimant""",
    ),
    (
        "GP RECORDS - RATHMORE MEDICAL PRACTICE (EXTRACT 1)",
        """Patient: Margaret Doyle, DOB 02/06/1971

14 March 2023 - Consultation, Dr Aoife Brennan
Patient first reported lower back pain, onset approximately two weeks ago after lifting boxes while moving house. Pain localised to the lumbar region, no radiation. Examination: paraspinal tenderness L4-L5, straight leg raise negative bilaterally. Plan: prescribed naproxen 500 mg twice daily with food. Advised to remain active. Review in six weeks if not settling.

02 April 2023 - Telephone consultation, Dr Aoife Brennan
Pain unchanged. Continuing naproxen. Sleep mildly disturbed. Advised paracetamol 1 g as required in addition. Review arranged.""",
    ),
    (
        "GP RECORDS - RATHMORE MEDICAL PRACTICE (EXTRACT 2)",
        """Patient: Margaret Doyle, DOB 02/06/1971

28 April 2023 - Consultation, Dr Aoife Brennan
Lower back pain worsening, now with intermittent aching into the left buttock. Naproxen giving partial relief only. Added codeine phosphate 30 mg up to four times daily. Discussed referral for physiotherapy; patient agreeable.

16 June 2023 - Consultation, Dr Aoife Brennan
Symptoms persist. Referral made to Lagan Valley Physiotherapy Clinic for assessment and management. Codeine continued. Patient advised on posture and graded activity.

01 September 2023 - Consultation, Dr Aoife Brennan
Physiotherapy completed six sessions with limited benefit. New symptom: shooting pain radiating down the left leg to the calf. Straight leg raise positive on the left at 40 degrees. Referral to Orthopaedic Outpatients, St Colmcille's General Hospital.""",
    ),
    (
        "REFERRAL LETTER",
        """Rathmore Medical Practice
14 Glenview Road

04 September 2023

To: Orthopaedic Outpatients, St Colmcille's General Hospital
Re: Margaret Doyle, DOB 02/06/1971

Dear Colleagues,

I would be grateful if you would see this 52-year-old woman with a six-month history of lower back pain following a lifting incident in March 2023. Despite naproxen, codeine phosphate, and a completed course of physiotherapy at Lagan Valley Physiotherapy Clinic, her symptoms have progressed and she now describes left-sided radicular pain to the calf with a positive straight leg raise at 40 degrees on the left.

I would value your assessment regarding further investigation and management.

Yours faithfully,
Dr Aoife Brennan, MB BCh""",
    ),
    (
        "PHYSIOTHERAPY INITIAL ASSESSMENT",
        """Lagan Valley Physiotherapy Clinic

Patient: Margaret Doyle
Date of assessment: 03 July 2023
Therapist: Fiona Marsh, MCSP

Presenting complaint: lower back pain since March 2023, now with left buttock ache. Referred by Dr Brennan for physiotherapy assessment and management.

Objective findings: reduced lumbar flexion (fingertips to mid-shin), paraspinal muscle guarding L3-L5, left hip flexor tightness. Neurological screen unremarkable at assessment.

Plan: six sessions - manual therapy, core stability programme, home exercise plan. First treatment session delivered today.""",
    ),
    (
        "PHYSIOTHERAPY PROGRESS NOTE",
        """Lagan Valley Physiotherapy Clinic

Patient: Margaret Doyle
Date: 21 August 2023
Therapist: Fiona Marsh, MCSP

Sixth and final physiotherapy session of the referred course. Patient reports approximately 20 per cent improvement in baseline lumbar pain but new intermittent shooting pain into the left leg over the past fortnight, worse with prolonged sitting.

Objective: lumbar flexion mildly improved. Left straight leg raise now reproducing leg symptoms at approximately 45 degrees.

Outcome: course complete. In view of the new radicular features, patient advised to return to her GP for review regarding onward referral. Discharge letter sent to Dr Brennan.""",
    ),
    (
        "MRI LUMBAR SPINE REPORT",
        """Ulster Imaging Centre

Patient: Margaret Doyle, DOB 02/06/1971
Examination: MRI lumbar spine (sagittal and axial T1/T2)
Date of examination: 26 September 2023
Reporting radiologist: Dr Priya Nathan, FRCR

Clinical details: six-month history of lower back pain with left-sided radicular symptoms. Query disc pathology.

Findings: normal vertebral alignment. Disc desiccation at L4/L5 with a moderate left paracentral disc protrusion contacting and displacing the traversing left L5 nerve root. Mild associated canal narrowing. No significant abnormality at other levels. Conus normal.

Impression: moderate left paracentral L4/L5 disc protrusion with left L5 nerve root compromise, concordant with the clinical presentation.""",
    ),
    (
        "HOSPITAL ADMISSION AND OPERATION NOTE",
        """St Colmcille's General Hospital

Patient: Margaret Doyle, DOB 02/06/1971
Date of admission: 15 November 2023
Consultant: Mr Conor Havelin, FRCS (Tr & Orth)

Admitted electively for surgical management of left L4/L5 disc protrusion after failed conservative management (analgesia and physiotherapy) and concordant MRI findings reported 26 September 2023.

Operation (15 November 2023): left L4/L5 microdiscectomy under general anaesthesia. Sequestrated disc fragment removed; L5 nerve root decompressed and mobile at closure. Estimated blood loss minimal. No intraoperative complications.

Post-operative course: neurologically intact, mobilising independently with physiotherapy input by day two. Wound clean and dry.""",
    ),
    (
        "DISCHARGE SUMMARY",
        """St Colmcille's General Hospital

Patient: Margaret Doyle, DOB 02/06/1971
Date of discharge: 18 November 2023
Responsible consultant: Mr Conor Havelin, FRCS (Tr & Orth)

Diagnosis: left L4/L5 disc protrusion with L5 radiculopathy.
Procedure: left L4/L5 microdiscectomy on 15 November 2023.

Medication changes on discharge:
- codeine phosphate STOPPED
- naproxen STOPPED
- amitriptyline 10 mg at night STARTED for residual neuropathic symptoms
- paracetamol 1 g up to four times daily as required CONTINUED

Follow-up: wound review with practice nurse at two weeks; orthopaedic outpatient review with Mr Havelin at six weeks; outpatient physiotherapy referral for post-operative rehabilitation.

Advice given: no heavy lifting for six weeks; graded return to activity.""",
    ),
    (
        "WITNESS STATEMENT OF THE CLAIMANT",
        """Statement of Margaret Doyle
Dated: 09 February 2024

1. I make this statement in support of my claim. The facts stated are true to the best of my knowledge and belief.

2. I first hurt my back in early March 2023 while lifting boxes during a house move. I saw Dr Brennan at Rathmore Medical Practice on 14 March 2023, who prescribed anti-inflammatory medication.

3. Over the following months the pain worsened despite tablets and six sessions of physiotherapy at Lagan Valley Physiotherapy Clinic. By late summer the pain was shooting down my left leg.

4. An MRI scan in September 2023 showed a disc problem in my lower back. Mr Havelin performed an operation on 15 November 2023 at St Colmcille's General Hospital, after which the leg pain substantially improved.

5. I continue to take a low-dose nightly medication and remain cautious with lifting. I have returned to work on reduced duties.

Signed: M. Doyle""",
    ),
]


def draw_page(c: canvas.Canvas, page_number: int, title: str, body: str) -> None:
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)

    c.setFont("Helvetica", 11)
    y = PAGE_H - MARGIN - 28
    for paragraph in body.split("\n"):
        if not paragraph.strip():
            y -= 8
            continue
        for line in simpleSplit(paragraph, "Helvetica", 11, BODY_WIDTH):
            c.drawString(MARGIN, y, line)
            y -= 15

    c.setFont("Helvetica", 9)
    c.drawCentredString(PAGE_W / 2, 30, f"Page {page_number} of {TOTAL_PAGES}")
    c.showPage()


def build_pdf(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    for page_number, (title, body) in enumerate(PAGES, start=1):
        draw_page(c, page_number, title, body)
    c.save()
    print(f"Wrote {out_path} ({len(PAGES)} pages)")


if __name__ == "__main__":
    build_pdf(OUT)
