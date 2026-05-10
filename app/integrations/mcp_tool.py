TOOL_SCHEMA = {
    "name": "extract_meeting_knowledge",
    "description": "Extract decisions, tasks and responsible people from meeting transcripts",
    "input_schema": {
        "type": "object",
        "properties": {
            "transcript": {
                "type": "string"
            }
        },
        "required": ["transcript"]
    }
}

from app.services.meeting_pipeline import process_meeting

def execute_tool(arguments: dict):
    transcript = arguments["transcript"]

    return process_meeting(transcript)