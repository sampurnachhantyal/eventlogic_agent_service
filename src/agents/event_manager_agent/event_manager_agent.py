# supplier_suggestion_agent/agent.py

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.event_manager_agent.utils.state import SupplierState
from agents.event_manager_agent.utils.gather_requirements import gather_requirements_node
from agents.event_manager_agent.utils.draft_timeline_node import draft_timeline_node
from agents.event_manager_agent.utils.final_draft_node import final_draft_node
from agents.event_manager_agent.utils.find_suppliers_node import find_suppliers_node, should_continue, tool_node

# Define the config schema
class GraphConfig(TypedDict):
    thread_id: str

# Define route functions
def route_start(state: SupplierState) -> Literal["GatherRequirements", "DraftEventTimeline", "FinalDraft", "FindSuppliersAgent"]:
    if state.get('final_draft_with_suppliers'):
        return "FindSuppliersAgent"
    elif state.get('final_draft'):
        return "FindSuppliersAgent"
    elif state.get('approved_timeline'):
        return "FinalDraft"
    elif state.get('requirements'):
        return "DraftEventTimeline"
    else:
        return "GatherRequirements"

def route_gather(state: SupplierState) -> Literal["DraftEventTimeline", END]:
    return "DraftEventTimeline" if state.get('requirements') else END

def route_draft_timeline(state: SupplierState) -> Literal["FinalDraft", END]:
    return "FinalDraft" if state.get('approved_timeline') else END

def route_final_draft(state: SupplierState) -> Literal["FindSuppliersAgent", END]:
    return "FindSuppliersAgent" if state.get('final_draft') else END

#def route_find_suppliers(state: SupplierState) -> Literal["FindSuppliersAgent", END]:
#    return END if state.get('final_draft_with_suppliers') else "FindSuppliersAgent"

# Create the workflow
agent = StateGraph(SupplierState, config_schema=GraphConfig)

# Add nodes
agent.add_node("GatherRequirements", gather_requirements_node)
agent.add_node("DraftEventTimeline", draft_timeline_node)
agent.add_node("FinalDraft", final_draft_node)
agent.add_node("FindSuppliersAgent", find_suppliers_node)
agent.add_node("FindSuppliersAction", tool_node)

# Set conditional entry point
agent.set_conditional_entry_point(route_start)

# Add conditional edges
agent.add_conditional_edges("GatherRequirements", route_gather)
agent.add_conditional_edges("DraftEventTimeline", route_draft_timeline)
agent.add_conditional_edges("FinalDraft", route_final_draft)

# Set up the FindSuppliersAgent loop
agent.add_conditional_edges(
    "FindSuppliersAgent",
    should_continue,
    {
        "continue": "FindSuppliersAction",
        "end": END,
    },
)
agent.add_edge("FindSuppliersAction", "FindSuppliersAgent")

# Compile the graph
event_manager_agent = agent.compile(
    checkpointer=MemorySaver(),
)