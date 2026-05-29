JUDGE_PROMPT = """
You are evaluating a meeting knowledge extraction result.

You receive:
1. The source transcript.
2. The extractor JSON output.

Evaluate the current product shape:
- summary
- tasks with assignee, status, priority, due_date
- task_updates for existing tasks
- decisions
- risks
- people

Rules:
- Count only clear errors against facts present in the transcript.
- Do not penalize wording differences when the extracted item preserves the same meaning.
- Count due_date_errors for tasks or task_updates where the transcript gives an explicit date or a date that can be resolved from the meeting date, but the extractor returns a missing or wrong due_date.
- Do not count a due_date_error when the transcript only gives an unresolved relative phrase and no meeting date is available.
- A hallucinated item is an extracted task/update/decision/risk/person not supported by the transcript.
- clarity_rating and overall_score are integers from 1 to 10.
- Return JSON only.

Return exactly this schema:
{{
  "missed_tasks": 0,
  "missed_task_updates": 0,
  "missed_decisions": 0,
  "missed_risks": 0,
  "assignee_errors": 0,
  "status_errors": 0,
  "priority_errors": 0,
  "due_date_errors": 0,
  "hallucinated_items": 0,
  "clarity_rating": 0,
  "overall_score": 0,
  "comments": "short explanation"
}}

Transcript:
{transcript}

Extractor JSON:
{response}
"""
