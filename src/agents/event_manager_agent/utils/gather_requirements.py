# supplier_suggestion_agent/utils/gather_requirements.py

import json
import logging
from typing import TypedDict
from datetime import datetime
from langchain_core.messages import RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI
from agents.event_manager_agent.utils.state import SupplierState

# Set up logging
logger = logging.getLogger(__name__)

current_date = datetime.now().strftime("%Y-%m-%d")

# Define the prompt
gather_prompt_template = """
You are a friendly and professional assistant helping users plan events by gathering their requirements.

The current date is {current_date}.

Engage the user in a conversational manner to collect their event details, asking one questions at a time.

Once you have gathered all the **mandatory** information, call the `Build` tool with the collected requirements.

**Mandatory event requirements:**
- event_type ()
- start_date
- end_date
- participants
- location

**Optional event requirements:**
- event_start_time (start time for the event)
- event_end_time (end time for the event, not day end time)
- overnight_guests (only ask if the event is overnight)
- event_contents (a list of event content items, each with associated parts)
- additional_requirements (do not ask for this, but add anything of relevance the user has provided for the event planning)
- preferred_suppliers (Note: This can include the location if it's also a preferred supplier)
- tasks (do not ask for this, but if user provides, include it)

If any information is unclear, feel free to ask the user for clarification.

Do not ask unnecessary questions or repeatedly confirm information already provided. If user gives indication of wanting a speedy process, treat it as option G. And you have the mandatory details, provide a suggestion for the event directly.

Each content can have multiple parts and preferred suppliers for the procurement process.
Think logically and if user suggests a conference, you will need to add "Content" as Conference Venue and "Parts" as conference room, breakout rooms, coffee break etc. You can ask for preferred suppliers for each part if necessary.
If it's a Kickoff event with accommodation and conference, you will need to add "Content" as Conference Hotel and "Parts" as single rooms, double rooms, conference room.. and so on.
IMPORTANT: Group parts that are likely to be procured from the same supplier under the same content.

With your responses include after your message a set of options for the user to choose from. Think of the most logical options based on the context of the question, the requirements and your expertise in event planning and the users previous responses.
See it as you are guiding yourself to creating the optimal event with user feedback along the way. The questions and options shoud be creative not just questions leading to yes, no options even if its also fine when necessary. Provide the options in a clear and concise manner. The options can be a longer sentense if appropriate. [[A, option A], [B, Option B], [C, Option C]] or [[A, Yes], [B, No]] or [[A, full sentence suggestion], [B, full sentence suggestion], [C..]].

Example. When asking for the event type, you could provide options like: [[A, Kickoff], [B, Conference] [C, Meeting], [D, Party], [E, Wedding], [F, Other]]. Always provide a last option [G, Impress me] option, where you will answer for the user as per the best of your ability. Do not ask for the event type if it's already provided.
Do not ask for ask for details again if it's already provided by the user. Then adapt the options to the event type and other details you have.
If option G, ask in a fun way if they can give you something to work with and explain shortly that the more you get the better with a funny anecdotal example of how wrong it can go when the customer and event planner are not aligned or planner has to few details. Details you will need, eg purpose, participants, location restrictions and date etc. then you to the best of your ability provide direclty a suggestion for the event and provide the entire requirements and optional requirements as well as the tasks for the event for the user to approve or adjust. Be creative and think of the best event for the user based on the information you have. DIRECTLY PROVIDE A SUGGESTION FOR THE EVENT. NO MORE QUESTIONS.
Length of event: [[A, 1 day], [B, 2 days], [C, 3 days], [D, 4 days]..
Location: option could be a list of cities or venues based on user language.
Start time, end time: could be a list of times or time ranges based on the event type, user preferences or your expertise.

IMPORTANT, you can gather supplier preferences as well as if user wants suggestions for suppliers, but DO NOT suggest specific suppliers in this step. As this will be done in a later step and you can say we will get to that soon.

When calling the `Build` tool, provide the requirements in a structured JSON format like so:

Build({{
  "requirements": {{
    "event_type": "...",
    "start_date": "...",
    "end_date": "...",
    "participants": "...",
    "location": "...",
    "overnight_guests": "...",
    "event_contents": [
      {{ "content": "...", "parts": ["...", "..."], "preferred_suppliers": ["...", "..."] }},
      {{ "content": "...", "parts": ["...", "..."], "preferred_suppliers": ["...", "..."] }}
    ],
    "additional_requirements": "...",
    "tasks": "..."
  }}
}})
"""

# Format the prompt with the current date
gather_prompt = gather_prompt_template.format(current_date=current_date)

# Define the Build tool schema
class Build(TypedDict):
    requirements: str

def format_requirements_table(requirements: str) -> str:
    try:
        # Parse the requirements string into a dictionary
        req_dict = json.loads(requirements)
        
        # Define the table structure
        table = "| Requirement | Value |\n|-------------|-------|\n"
        
        # Add each requirement to the table
        for key, value in req_dict.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)  # Convert complex structures to JSON string
            table += f"| {key} | {value} |\n"
        
        return table
    except json.JSONDecodeError as e:
        return f"Error parsing requirements: {str(e)}\nRaw requirements: {requirements}"

def gather_requirements_node(state: SupplierState, config):
    messages = [
        {"role": "system", "content": gather_prompt}
    ] + state['messages']

    # Get the LLM model (you might need to adjust this based on your config structure)
    llm = ChatOpenAI(temperature=0.8, model_name="gpt-4o")
    model = llm.bind_tools([Build])

    try:
        response = model.invoke(messages)

        if len(response.tool_calls) == 0:
            return {"messages": [AIMessage(content=response.content)]}
        else:
            requirements = response.tool_calls[0]['args']['requirements']
            
            # Ensure requirements is a JSON string
            if not isinstance(requirements, str):
                requirements = json.dumps(requirements)
            
            # Format requirements as a table
            formatted_requirements = format_requirements_table(requirements)
            
            delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
            return {
                "requirements": requirements,  # Keep the original string for further processing if needed
                "formatted_requirements": formatted_requirements,  # Add the formatted table
                "messages": delete_messages + [AIMessage(content=f"Here are the gathered requirements:\n\n{formatted_requirements}")]
            }
    except Exception as e:
        logger.exception(f"An error occurred while processing the model response: {str(e)}")
        return {"messages": [AIMessage(content=f"An error occurred: {str(e)}")]}
    

