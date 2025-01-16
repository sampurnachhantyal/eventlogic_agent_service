import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, RemoveMessage
from agents.event_manager_agent.utils.state import SupplierState
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CONFIG = {
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "max_retries": 3,
    "cache_size": 128,
}

def ensure_json_serializable(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj)
    except TypeError:
        return str(obj)

def extract_timeline_from_response(content: str) -> List[Dict[str, Any]]:
    try:
        # Remove the finalization keyword and code block markers before parsing
        content = content.replace("--timeline_finalized--", "").strip()
        content = content.split("```json")[1].split("```")[0].strip()
        # Parse the JSON content
        timeline = json.loads(content)
        
        # Ensure the timeline is a list
        if not isinstance(timeline, list):
            raise ValueError("Timeline is not a list")
        
        # Validate the structure of each item in the timeline
        for item in timeline:
            if "date" in item and "events" in item:
                # Validate each event in the day
                for event in item["events"]:
                    if not all(key in event for key in ["time", "name", "duration"]):
                        raise ValueError(f"Invalid event structure in {item}")
            elif "additional_preferences" in item:
                if not isinstance(item["additional_preferences"], list):
                    raise ValueError("additional_preferences is not a list")
            else:
                raise ValueError(f"Invalid item structure: {item}")
        
        return timeline
    except (json.JSONDecodeError, IndexError, ValueError) as e:
        # Handle parsing error
        logger.error(f"Failed to parse timeline from response: {e}")
        return []

current_date = datetime.now().strftime("%Y-%m-%d")

# Define the prompt
draft_timeline_prompt = """
You are a friendly and professional assistant helping users plan event timelines.
The current date is {current_date}.
Event Requirements: {requirements}

Keep conversation in Swedish if user doesn't specify language.

Given the event requirements, and to the best of your abilities, present a timeline to the user.

Ask only one qustion at a time.

If conference in the requirements, and not already provided in the detials you have, eg. ask for the following: Do you want to provide what type of conference it is? [[A, Internal conference], [B, Conference with invited clients], [C, Product launch conference], [D, Other type of conference], [E, Skip for now]]. I can suggest a timeline based on the type of conference you choose. No rush though, we can do it later.
Do the same for other event types, but adapt the question to the event type.

With your responses include after your message a set of options for the user to choose from. Think of the most logical options based on the context of the question, the requirements and your expertise in event planning and the users previous responses.
Think of the options as a way to allow the user to quickly answer your questions. Provide the options in a clear and concise manner. The options can be a longer sentense if appropriate. [[A, option A], [B, Option B], [C, Option C]] or [[A, Yes], [B, No]] or [[A, full sentence suggestion], [B, full sentence suggestion], [C..]].

The final output should be a JSON array of objects, where each object represents a day and contains an array of events for that day.
Example format:
[
    {{
        "date": "2024-09-22",
        "events": [
            {{"time": "09:00", "name": "Registration", "duration": "1 hour"}},
            {{"time": "10:00", "name": "Opening Keynote", "duration": "1 hour"}}
        ]
    }},
    {{
        "date": "2024-09-23",
        "events": [
            {{"time": "09:30", "name": "Workshop Session", "duration": "2 hours"}},
            {{"time": "12:00", "name": "Networking Lunch", "duration": "1.5 hours"}}
        ]
        "additional_gathered_preferences": "..."
    }}
    {{
        "additional_preferences": [
            "Vegetarian options required for catering",
            "Need projectors in all rooms",
            "Inerested in supplier X for the teambuilding activity", etc.
        ]
    }}
]
Have a dialog with the user with suggestions to iterate the timeline together until they approve it. Once the timeline is finalized, output it with the structure above in strict JSON and include the phrase '--timeline_finalized--' after.
"""

def format_timeline_table(timeline: str) -> str:
    try:
        timeline_list = json.loads(timeline)
        table = "| Date | Time | Event | Duration |\n|------|------|-------|----------|\n"
        for day in timeline_list:
            if 'date' in day and 'events' in day:
                date = day["date"]
                for event in day["events"]:
                    table += f"| {date} | {event['time']} | {event['name']} | {event['duration']} |\n"
        return table
    except json.JSONDecodeError as e:
        return f"Error parsing timeline: {str(e)}\nRaw timeline: {timeline}"

def draft_timeline_node(state: SupplierState, config):
    messages = [
        {"role": "system", "content": draft_timeline_prompt.format(current_date=current_date, requirements=json.dumps(state.get("requirements", {})))}
    ] + state['messages']

    llm = ChatOpenAI(temperature=0.7, model_name=config.get("model_name", "gpt-4o-mini"))

    try:
        response = llm.invoke(messages)
        
        if "--timeline_finalized--" in response.content:
            timeline = extract_timeline_from_response(response.content)
            
            # Separate the timeline and additional preferences
            additional_preferences = next((item['additional_preferences'] for item in timeline if 'additional_preferences' in item), [])
            timeline_events = [item for item in timeline if 'date' in item]
            
            state['approved_timeline'] = {"timeline": timeline_events, "additional_preferences": additional_preferences}
            
            formatted_timeline = format_timeline_table(json.dumps(timeline_events))
            delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
            
            return {
                "approved_timeline": state['approved_timeline'],
                "formatted_timeline": formatted_timeline,
                "messages": delete_messages + [AIMessage(content=f"Here's the final timeline for your event:\n\n{formatted_timeline}\n\nAdditional preferences:\n" + "\n".join(f"- {pref}" for pref in additional_preferences) + "\n\nThe timeline has been approved and we're moving on to the next step in planning your event.")]
            }
        else:
            return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.exception(f"An error occurred while processing the model response: {str(e)}")
        return {"messages": [AIMessage(content=f"An error occurred: {str(e)}")]}
