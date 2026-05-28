SILVER_TRUTH_PROMPT = """
You are creating persistent silver-standard labels for a meeting transcript extraction benchmark.

The transcript is the only source of truth. Extract only facts explicitly present in it.
Be exhaustive, but do not invent details. If a due date, assignee, status, or priority is unclear, use null or the safest explicit value.

Return one valid JSON object with this exact shape:
{
  "schema_version": 1,
  "summary": "...",
  "tasks": [
    {
      "description": "...",
      "assignee": "...",
      "status": "todo|in_progress|done",
      "priority": "low|medium|high",
      "due_date": "YYYY-MM-DD or null",
      "evidence": "short quote or paraphrase from transcript"
    }
  ],
  "task_updates": [
    {
      "description": "...",
      "assignee": "... or null",
      "status": "todo|in_progress|done",
      "due_date": "YYYY-MM-DD or null",
      "evidence": "short quote or paraphrase from transcript"
    }
  ],
  "decisions": [
    {
      "description": "...",
      "evidence": "short quote or paraphrase from transcript"
    }
  ],
  "risks": [
    {
      "description": "...",
      "evidence": "short quote or paraphrase from transcript"
    }
  ],
  "people": [
    {
      "name": "...",
      "role": "... or null"
    }
  ],
  "notes": [
    "Ambiguities or labeling assumptions."
  ]
}

Rules:
- Use Russian text for Russian transcripts.
- Normalize task statuses to todo, in_progress, or done.
- Normalize priorities to low, medium, or high.
- Use ISO dates when the transcript provides enough information to resolve them.
- Keep descriptions concise but specific enough for fuzzy matching.
- Evidence should be short and useful for later human review.
- Return JSON only.

Transcript file: {file_name}

Transcript:
{transcript}
""".strip()
