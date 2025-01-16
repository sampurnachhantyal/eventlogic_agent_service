# supplier_suggestion_agent/utils/final_draft_node.py

import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, RemoveMessage
from agents.event_manager_agent.utils.state import SupplierState
import logging

# Set up logging
logger = logging.getLogger(__name__)

current_date = datetime.now().strftime("%Y-%m-%d")

# Define the prompt
final_draft_prompt = """
You are a genius event expert helping users finalize their event plans and preparing the final draft for the procurement process agent.
The current date is {current_date}.

Here are the outcomes from a discussion with a user to create their dream event:

Requirements: {requirements}
Additional Requirements: {additional_requirements}
Timeline: {timeline}

Check for language in earlier discussions. Proceed with the same language.

These might not be fully aligned. Your job is to create a comprehensive final draft that includes:
1. Updated requirements, based on all the details you received (maintaining the original structure)
Your most important task is aligning the details for the event and providning the final draft. Updating the content and parts if they are not already. Content can be anything from a conference venue to teambuilding activity to a restaurant. Parts are the specific items that make up the content and have amount in PIECES or PEOPLE. 
Each content can have multiple parts and preferred suppliers for the procurement process. 
For example, a conference venue might have parts like conference room, breakout rooms, coffee break etc. 
IMPORTANT: Each content and its parts is something that needs to be procured or arranged for the event from one supplier. eg for a conference the Venue will most likely provide rooms, av equipment, meals etc. If its not structured this way update it.
The content and parts should be aligned with the timeline, even though parts can be spread out over multiple days and some might be timeless. All parts and timeline events need to be fully aligned by name. If amount PEOPLE assume same amount as participants. Use PIECES where PEOPLE is not applicable and for items. 
2. Final timeline (maintaining the original structure), aligned with the updated requirements.

Provide this initial draft in a nice human readable format for the user to review. Should NOT be in JSON format.

Ask if they wish to get your professional feedback. If so act as the top event planning expert and provide your feedback on the draft. Do not suggest specific suppliers as this will be done in a later step.
Providing choices below: [[A, Yes, I want your feedback], [B, No, I'm happy with the draft]].

Provide the feedback in numbered points, each point should be a suggestion for improvement, so user can easily pick and choose what to change. Explain this to the user.
Below your feedback provide the options as buttons, [[Include option 1], [Include option 2].. etc and finally [Include all suggestions].

Have a dialog with the user, presenting your guidance and suggestions for improvements along the way. Iterate on the final draft together until user is satisfied.

REMEMBER your final goal is to output the final draft in JSON format with the structure below always.

The final draft should follow this structure:

{{
  "final_draft_approved": {{
    "requirements": {{
      "event_type": "Kickoff", 
      "start_date": "2024-11-10", 
      "end_date": "2024-11-11",
      "event_start_time": "10:00", 
      "event_end_time": "15:30",
      "participants": "20", 
      "location": "Gothenburg", 
      "overnight_guests": "Yes",
      "event_contents": [
        {{
          "content": "Conference Hotel",
          "parts": [
            {{ "name": "Conference room Day 1", "amount": 1, "amount_type": "PIECES", "timeless": false, "date": "2024-11-10", "time": "10:00", "duration_hours": "5,5" }},
            {{ "name": "Conference room Day 2", "amount": 1, "amount_type": "PIECES", "timeless": false, "date": "2024-11-11", "time": "09:00", "duration_hours": "6,5" }},
            {{ "name": "Breakout rooms Day 1", "amount": 2, "amount_type": "PIECES", "timeless": false, "date": "2024-11-10", "time": "13:30", "duration_hours": "2,0" }},
            {{ "name": "Workshop room Day 2", "amount": 1, "amount_type": "PIECES", "timeless": false, "date": "2024-11-11", "time": "13:00", "duration_hours": "2,0" }},
            {{ "name": "Single rooms", "amount": 10, "amount_type": "PIECES", "timeless": false, "date": "2024-11-10", "time": "14:00", "duration_hours": "24,0" }},
            {{ "name": "Double rooms", "amount": 5, "amount_type": "PIECES", "timeless": false, "date": "2024-11-10", "time": "14:00", "duration_hours": "24,0" }}
          ],
          "potential_suppliers": [
            {{
              "supplier_name": "Gothenburg Conference Center",
              "potential_supplier_id": 1044,
              "el_supplier_id": 2238295
            }}
          ]
        }},
        {{
          "content": "Activity",
          "parts": [
            {{ "name": "Team-building activity", "amount": 20, "amount_type": "PEOPLE", "timeless": false, "date": "2024-11-10", "time": "16:00", "duration_hours": "2,0" }}
          ],
          "potential_suppliers": [
            {{
              "supplier_name": "Adventure Gothenburg",
              "potential_supplier_id": 3001,
              "el_supplier_id": 4001001
            }}
          ]
        }},
        {{
          "content": "Restaurant",
          "parts": [
            {{ "name": "Three course dinner", "amount": 20, "amount_type": "PEOPLE", "timeless": false, "date": "2024-11-10", "time": "19:00", "duration_hours": "2,0" }},
            {{ "name": "Wine package", "amount": 20, "amount_type": "PEOPLE", "timeless": false, "date": "2024-11-10", "time": "19:00", "duration_hours": "2,0" }}
          ],
          "potential_suppliers": [
            {{
              "supplier_name": "Gourmet Restaurant",
              "potential_supplier_id": 4001,
              "el_supplier_id": 5001001
            }}
          ]
        }}
      ],
      "additional_requirements": "Need suggestions for suppliers and activities. Prefer eco-friendly options when possible.",
      "tasks": [
        "Organize a team-building activity",
        "Arrange group dining",
        "Coordinate transportation between venues if needed"
      ]
    }},
    "timeline": [
      {{
        "date": "2024-11-10",
        "events": [
          {{
            "time": "10:00",
            "name": "Registration",
            "duration_hours": "1,0",
            "location": {{
              "name": "Conference Hotel",
              "coordinates": {{ "latitude": 57.70887, "longitude": 11.97456 }}
            }}
          }},
          {{
            "time": "11:00",
            "name": "Opening session in conference room",
            "duration_hours": "1,5",
            "location": {{
              "name": "Conference room",
              "coordinates": {{ "latitude": 57.70887, "longitude": 11.97456 }}
            }}
          }},
          {{
            "time": "12:30",
            "name": "Lunch",
            "duration_hours": "1,0",
            "location": {{
              "name": "Dining Area",
              "coordinates": {{ "latitude": 57.70887, "longitude": 11.97456 }}
            }}
          }},
          // ... more events for Day 1 ...
        ]
      }},
      {{
        "date": "2024-11-11",
        "events": [
          {{
            "time": "08:00",
            "name": "Breakfast at the hotel",
            "duration_hours": "1,0",
            "location": {{
              "name": "Dining Area",
              "coordinates": {{ "latitude": 57.70887, "longitude": 11.97456 }}
            }}
          }},
          {{
            "time": "09:00",
            "name": "Continuation of conference sessions",
            "duration_hours": "3,0",
            "location": {{
              "name": "Conference room",
              "coordinates": {{ "latitude": 57.70887, "longitude": 11.97456 }}
            }}
          }},
          // ... more events for Day 2 ...
        ]
      }}
    ]
  }}
}}

"""

def extract_final_draft_from_response(content: str) -> Dict[str, Any]:
    try:
        # Remove the finalization keyword and code block markers before parsing
        content = content.replace("--final_draft_approved--", "").strip()
        content = content.split("```json")[1].split("```")[0].strip()
        # Assuming the final draft is included in the response content as JSON
        return json.loads(content)
    except (json.JSONDecodeError, IndexError) as e:
        # Handle parsing error
        logger.error(f"Failed to parse final draft from response: {e}")
        return {}

def format_final_draft(final_draft: Dict[str, Any]) -> str:
    formatted = "Final Event Draft:\n\n"
    
    if 'final_draft' in final_draft:
        final_draft = final_draft['final_draft']
    
    if 'requirements' in final_draft:
        formatted += "Updated Requirements:\n"
        for key, value in final_draft['requirements'].items():
            if key == 'event_contents':
                formatted += f"- {key}:\n"
                for content in value:
                    formatted += f"  - {content['content']}:\n"
                    formatted += "    Parts:\n"
                    for part in content['parts']:
                        formatted += f"      - {part['name']} (Amount: {part['amount']} {part['amount_type']})\n"
                    formatted += f"    Preferred Suppliers: {', '.join(content['preferred_suppliers'])}\n"
            else:
                formatted += f"- {key}: {value}\n"
    
    if 'timeline' in final_draft:
        formatted += "\nFinal Timeline:\n"
        for day in final_draft['timeline']:
            formatted += f"Date: {day['date']}\n"
            for event in day['events']:
                formatted += f"  - {event['time']}: {event['name']} ({event['duration']})\n"
    
    return formatted

def final_draft_node(state: SupplierState, config):
    messages = [
        {"role": "system", "content": final_draft_prompt.format(
            current_date=current_date,
            requirements=json.dumps(state.get("requirements", {})),
            additional_requirements=state.get("additional_requirements", ""),
            timeline=json.dumps(state.get("approved_timeline", {}))
        )}
    ] + state['messages']

    llm = ChatOpenAI(temperature=0.7, model_name=config.get("model_name", "gpt-4o-mini"))

    try:
        response = llm.invoke(messages)
        
        # Check if the response contains the finalization keyword
        if "final_draft_approved" in response.content:
            # Extract the final draft from the response
            final_draft = extract_final_draft_from_response(response.content)
            
            # Update the state with the approved final draft
            state['final_draft'] = final_draft
            
            # Format final draft
            formatted_final_draft = format_final_draft(final_draft)
            delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
            
            return {
                "final_draft": final_draft,  # This signals that the final draft is complete
                "formatted_final_draft": formatted_final_draft,
                "messages": delete_messages + [AIMessage(content=f"Here's the final draft for your event:\n\n{formatted_final_draft}\n\nThe final draft has been approved and we're moving on to the next step in planning your event.")]
            }
        else:
            return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.exception(f"An error occurred while processing the model response: {str(e)}")
        return {"messages": [AIMessage(content=f"An error occurred: {str(e)}")]}