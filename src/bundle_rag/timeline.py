"""Timeline extraction from medical bundle chunks."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dateutil import parser as date_parser
from rich.console import Console

console = Console()

# Date patterns (reusing from vectorstore.py but enhanced)
DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY
    (r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b', 'dmy'),
    (r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b', 'dmy_short'),
    # Ordinal dates: 23rd December 2016
    (r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', 'ordinal'),
    # Month DD, YYYY
    (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b', 'mdy'),
]

# Event type classification patterns
EVENT_TYPE_PATTERNS = {
    'consultation': ['consultation', 'seen by', 'review', 'appointment', 'surgery consultation', 'clinic', 'follow-up'],
    'test': ['mri', 'ct scan', 'x-ray', 'blood test', 'scan', 'results', 'imaging', 'ultrasound', 'ecg', 'echo'],
    'procedure': ['surgery', 'operation', 'procedure', 'injection', 'biopsy', 'excision', 'removal'],
    'admission': ['admission', 'admitted', 'inpatient', 'emergency', 'a&e', 'casualty', 'presented'],
    'discharge': ['discharge', 'discharged', 'home', 'sent home'],
    'prescription': ['prescribed', 'medication', 'started on', 'commenced', 'mg', 'capsule', 'tablet'],
}

# Medical section headers (reusing from ingest.py)
SECTION_HEADERS = [
    'HISTORY', 'EXAMINATION', 'DIAGNOSIS', 'TREATMENT', 'ASSESSMENT', 'PLAN',
    'IMPRESSION', 'FINDINGS', 'MEDICATIONS', 'ALLERGIES', 'VITALS', 'LABS',
    'IMAGING', 'CONSULTATION', 'DISCHARGE', 'ADMISSION', 'PROGRESS NOTE', 'CLINICAL NOTE'
]


def normalize_date(date_str: str) -> Optional[datetime]:
    """Parse various date formats to datetime using UK format preference.

    Validates that dates fall within reasonable medical record range (1950-2030).
    """
    try:
        # UK format preference (dayfirst=True)
        parsed = date_parser.parse(date_str, dayfirst=True, fuzzy=True)

        # Validate date is within reasonable medical record range
        # Medical records typically span 1950-2030
        if parsed.year < 1950 or parsed.year > 2030:
            return None

        # Handle 2-digit years more intelligently
        # If year is 00-30, assume 2000s; if 31-99, assume 1900s
        if parsed.year < 100:
            if parsed.year <= 30:
                parsed = parsed.replace(year=2000 + parsed.year)
            else:
                parsed = parsed.replace(year=1900 + parsed.year)

        return parsed
    except:
        return None


def extract_dates_from_text(text: str) -> list[dict]:
    """Extract all date mentions from text with context.

    Filters out common false positives like page numbers and boilerplate text.
    """
    dates = []
    text_lines = text.split('\n')

    # Patterns that indicate boilerplate/non-clinical text to skip
    SKIP_PATTERNS = [
        r'copyright',
        r'©',
        r'all rights reserved',
        r'page \d+',
        r'p\.\s*\d+',
        r'version \d+',
        r'document id',
        r'reference:?\s*\d+',
    ]

    for line_no, line in enumerate(text_lines):
        # Skip lines that match boilerplate patterns
        line_lower = line.lower()
        if any(re.search(pattern, line_lower) for pattern in SKIP_PATTERNS):
            continue

        for pattern, pattern_type in DATE_PATTERNS:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                parsed_date = normalize_date(date_str)

                if parsed_date:
                    # Get context around the date (the line it's on)
                    context_start = max(0, line_no - 1)
                    context_end = min(len(text_lines), line_no + 2)
                    context_lines = text_lines[context_start:context_end]
                    context = '\n'.join(context_lines).strip()

                    # Skip if context is too short (likely just a page number)
                    if len(context.strip()) < 20:
                        continue

                    dates.append({
                        'date': parsed_date,
                        'date_str': date_str,
                        'context': context,
                        'line_no': line_no,
                        'pattern_type': pattern_type
                    })

    return dates


def classify_event_type(text: str) -> str:
    """Classify event type based on keywords in text."""
    text_lower = text.lower()

    # Count keywords for each type
    scores = {}
    for event_type, keywords in EVENT_TYPE_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[event_type] = score

    # Return type with highest score
    if scores:
        return max(scores, key=scores.get)

    # Default to consultation if no clear type
    return 'consultation'


def extract_provider_name(text: str) -> Optional[str]:
    """Extract doctor/provider name from text."""
    # Try to find Dr. Name patterns
    dr_pattern = r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    match = re.search(dr_pattern, text)
    if match:
        return f"Dr {match.group(1)}"

    # Try consultant/GP patterns
    provider_pattern = r'(?:Consultant|GP|Surgeon)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    match = re.search(provider_pattern, text)
    if match:
        return match.group(1)

    return None


def extract_location(text: str) -> Optional[str]:
    """Extract hospital/clinic location from text."""
    # Common Northern Ireland hospitals
    hospitals = [
        'Ulster Hospital', 'Royal Victoria Hospital', 'Belfast City Hospital',
        'Mater Hospital', 'Causeway Hospital', 'Antrim Area Hospital',
        'Altnagelvin Hospital', 'Craigavon Hospital', 'South West Acute Hospital'
    ]

    text_lower = text.lower()
    for hospital in hospitals:
        if hospital.lower() in text_lower:
            return hospital

    # Try generic patterns
    location_pattern = r'([A-Z][a-z]+\s+Hospital|[A-Z][a-z]+\s+Clinic)'
    match = re.search(location_pattern, text)
    if match:
        return match.group(1)

    return None


def generate_event_title(date_info: dict, event_type: str, provider: Optional[str], location: Optional[str]) -> str:
    """Generate a concise title for the event."""
    parts = []

    # Event type
    type_labels = {
        'consultation': 'Consultation',
        'test': 'Investigation',
        'procedure': 'Procedure',
        'admission': 'Hospital Admission',
        'discharge': 'Discharge',
        'prescription': 'Medication Change'
    }
    parts.append(type_labels.get(event_type, 'Clinical Event'))

    # Location or provider
    if location:
        parts.append(f" - {location}")
    elif provider:
        parts.append(f" with {provider}")

    return ''.join(parts)


def extract_clinical_events(chunk: dict) -> list[dict]:
    """Extract dated clinical events from a chunk."""
    text = chunk['text']
    page = chunk.get('page', 0)
    chunk_id = chunk.get('idx', 0)

    # Extract all dates from this chunk
    date_mentions = extract_dates_from_text(text)

    if not date_mentions:
        return []

    events = []

    # For each date, create an event
    for date_info in date_mentions:
        event_type = classify_event_type(date_info['context'])
        provider = extract_provider_name(date_info['context'])
        location = extract_location(text)

        # Generate description (first few lines of context)
        description = date_info['context'][:300]
        if len(date_info['context']) > 300:
            description += '...'

        event = {
            'date': date_info['date'].strftime('%Y-%m-%d'),
            'date_raw': date_info['date_str'],
            'date_confidence': 0.95,  # High confidence for pattern matches
            'event_type': event_type,
            'title': generate_event_title(date_info, event_type, provider, location),
            'description': description,
            'provider': provider,
            'location': location,
            'chunk_ids': [chunk_id],
            'pages': [page],
            'source_text_preview': date_info['context'][:200]
        }

        events.append(event)

    return events


def merge_duplicate_events(events: list[dict]) -> list[dict]:
    """Merge events that likely refer to the same clinical encounter."""
    if not events:
        return []

    # Sort by date first
    events.sort(key=lambda e: e['date'])

    merged = []
    current_group = [events[0]]

    for event in events[1:]:
        prev_event = current_group[-1]

        # Check if events should be merged (same date, similar context)
        same_date = event['date'] == prev_event['date']
        similar_pages = abs(event['pages'][0] - prev_event['pages'][0]) <= 2

        if same_date and similar_pages:
            # Merge chunk_ids and pages
            current_group.append(event)
        else:
            # Save merged event
            if len(current_group) > 1:
                # Combine information from group
                merged_event = current_group[0].copy()
                merged_event['chunk_ids'] = list(set([cid for e in current_group for cid in e['chunk_ids']]))
                merged_event['pages'] = sorted(list(set([p for e in current_group for p in e['pages']])))
                # Combine descriptions
                descriptions = [e['description'] for e in current_group]
                merged_event['description'] = ' | '.join(set(descriptions))[:500]
                merged.append(merged_event)
            else:
                merged.append(current_group[0])

            current_group = [event]

    # Don't forget last group
    if current_group:
        if len(current_group) > 1:
            merged_event = current_group[0].copy()
            merged_event['chunk_ids'] = list(set([cid for e in current_group for cid in e['chunk_ids']]))
            merged_event['pages'] = sorted(list(set([p for e in current_group for p in e['pages']])))
            descriptions = [e['description'] for e in current_group]
            merged_event['description'] = ' | '.join(set(descriptions))[:500]
            merged.append(merged_event)
        else:
            merged.append(current_group[0])

    return merged


def build_timeline(case_dir: Path) -> dict:
    """Build timeline from chunks.json."""
    console.print("[yellow]Building timeline...[/yellow]")

    case_dir = Path(case_dir)
    chunks_file = case_dir / "chunks.json"

    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks.json not found in {case_dir}")

    # Load chunks
    with open(chunks_file) as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    console.print(f"[yellow]Processing {len(chunks)} chunks...[/yellow]")

    # Extract events from all chunks
    all_events = []
    for i, chunk in enumerate(chunks):
        if i % 50 == 0:
            console.print(f"  Processed {i}/{len(chunks)} chunks")

        chunk_events = extract_clinical_events(chunk)
        all_events.extend(chunk_events)

    console.print(f"[yellow]Found {len(all_events)} raw events[/yellow]")

    # Merge duplicate events
    events = merge_duplicate_events(all_events)
    console.print(f"[yellow]Merged to {len(events)} unique events[/yellow]")

    # Add unique IDs
    for i, event in enumerate(events):
        event['id'] = f"evt_{i:03d}"

    # Calculate date range
    if events:
        dates = [datetime.strptime(e['date'], '%Y-%m-%d') for e in events]
        date_range = {
            'earliest': min(dates).strftime('%Y-%m-%d'),
            'latest': max(dates).strftime('%Y-%m-%d')
        }
    else:
        date_range = {'earliest': None, 'latest': None}

    timeline = {
        'generated_at': datetime.now().isoformat(),
        'date_range': date_range,
        'event_count': len(events),
        'events': events
    }

    # Save to file
    timeline_file = case_dir / "timeline.json"
    with open(timeline_file, "w") as f:
        json.dump(timeline, f, indent=2)

    console.print(f"[green]Timeline saved: {len(events)} events from {date_range['earliest']} to {date_range['latest']}[/green]")

    return timeline
