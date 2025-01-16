# supplier_suggestion_agent/utils/find_suppliers_node.py

import json
from typing import Dict, Any
from datetime import datetime
from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
import logging
from agents.event_manager_agent.utils.tools_utils import prepare_supplier_data
from agents.event_manager_agent.utils.tools import tools

# Set up logging
logger = logging.getLogger(__name__)

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if (model_name == "openai"):
        model = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
    elif (model_name == "anthropic"):
        model = ChatAnthropic(temperature=0.7, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

def extract_final_draft_with_suppliers(content: str) -> Dict[str, Any]:
    try:
        # Extract JSON from code block if present
        if "```json" in content and "```" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        else:
            # If not in code block, try to find JSON object
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                content = content[start:end]

        # Parse the JSON content
        parsed_content = json.loads(content)
        
        # Extract the final_draft_with_suppliers part
        if "final_draft_with_suppliers" in parsed_content:
            return parsed_content["final_draft_with_suppliers"]
        else:
            logger.warning("'final_draft_with_suppliers' key not found in parsed JSON")
            return {}
    except json.JSONDecodeError:
        logger.info("Response is not in JSON format")
        return {}

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def find_suppliers_node(state, config):
    current_date = datetime.now().strftime("%Y-%m-%d")
    final_draft = json.dumps(state.get("final_draft", {}))

    system_prompt = f"""You are an expert in pairing the best suppliers to send requests to for the user's event they are planning. Your task is to help find suitable suppliers for each content in the draft NOT for each part, based on the final draft of an event plan. For each event content. 
    The end goal is to find the best 4 suppliers for each content in the event for the coming procurement. You shall primarely use the fetch suppliers tool and tavily search tool to find the best suppliers for each content. DO NOT limit when searching for suppliers. But consolidate your suggestions to max 4 to the user.

ONLY SUGGEST SUPPLIERS FOR EACH CONTENT, NEVER FOR PARTS.

Keep conversation in Swedish if user doesn't specify language.
    
Your job is to work with the user to find up to max 4 suppliers that best match their requirements for each content in their event. Once you have gathered knowledge and suggested the best potential suppliers for each content and gotten the user's final approval for all contents, call the Build tool.

If changes have been made that will alter the final draft, update the requirements and timeline to reflect this when outputting the json. (Example: If the final draft says the teambuilding activity should be bowling, but you suggest suppliers that also have other activities and the user wants to replace bowling with shuffleboard, update the content and parts in the final draft, add the supplier as potential suppliers, and update the timeline if necessary.)
IMPORTANT! All timeline events must be represented as parts and ALL parts as timeline events. If any timeline events don't fit existing contents create content "Internal" and place them there. Eg any activities that should not be procured.
If amount PEOPLE assume same amount as participants. Use PIECES where PEOPLE is not applicable and for items. 
If a part does not have a specific time (i.e., it is not time-bound or doesn't require a time), set the `timeless` field to `true` in the JSON. Otherwise, set `timeless` to `false` if the part has a specific time.
Add location name and coordinates to the timeline events only if they are known in this stage of the planning. Eg. venue or hotel is not finalized yet, then do NOT add. If the user has only one supplier of interest for venue, then add the location name and coordinates to the timeline event.

Always refer to yourself as the Event Logic Assistant.

Event details: {final_draft}
The current date is {current_date}.

If user asks to create the event using the create event tool, YOU MUST ALWAYS ASK FOR USER EMAIL.

Please help find the best suppliers for each content in the event except for internal content. Go through each content one by one and discuss potential suppliers.

When you have gathered the user's final approval for all contents, output the final draft with suppliers with the structure below in strict JSON and the instructions above.


The final draft with suppliers MUST follow this structure:

{{
  "final_draft_with_suppliers": {{
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

    messages = state["messages"]
    
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    #print(f"Messages: {messages}")
    
    response = model.invoke(messages)
    
    # Initialize state updates with the new message
    state_updates = {"messages": [response]}

    # Check if the response contains the final draft with suppliers
    if "final_draft_with_suppliers" in response.content:
        # Extract the final draft with suppliers
        final_draft_with_suppliers = extract_final_draft_with_suppliers(response.content)

        # Update the state with the final draft with suppliers
        state_updates['final_draft_with_suppliers'] = final_draft_with_suppliers
    
    print(state.get("final_draft_with_suppliers", {}))
    return state_updates

# Define the function to execute tools
tool_node = ToolNode(tools)

