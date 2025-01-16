# supplier_suggestion_agent/utils/state.py

from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class SupplierData(TypedDict, total=False):
    supplier_name: str
    rating: str
    country_code: str
    town: str
    filtered_supplier_id: Optional[str]
    potential_supplier_id: str
    el_supplier_id: Optional[str]
    status: str
    match_status: str
    supplier_link: str
    el_admin_link: str

class PotentialSupplier(TypedDict):
    supplier_name: str
    potential_supplier_id: str
    el_supplier_id: Optional[str]

class EventContent(TypedDict):
    content: Optional[str]
    parts: Optional[List[str]]
    preferred_suppliers: Optional[List[str]]
    potential_suppliers: Optional[List[PotentialSupplier]]

class Requirements(TypedDict):
    event_type: str
    start_date: str
    end_date: str
    event_start_time: Optional[str]
    event_end_time: Optional[str]
    participants: str
    location: str
    overnight_guests: Optional[str]
    event_contents: Optional[List[EventContent]]
    additional_requirements: Optional[str]
    tasks: Optional[str]

class FinalDraft(TypedDict):
    requirements: Dict[str, Any]
    timeline: List[Dict[str, Any]]

class FinalDraftWithSuppliers(TypedDict):
    requirements: Dict[str, Any]
    timeline: List[Dict[str, Any]]

class TimelineEvent(TypedDict):
    time: str
    name: str
    duration: str

class DayTimeline(TypedDict):
    date: str
    events: List[TimelineEvent]

class MapPoint(TypedDict):
    name: str
    lat: float
    lng: float

class MapData(TypedDict):
    center: Dict[str, float]
    radius: int
    points_of_interest: List[MapPoint]

class SupplierRequirement(TypedDict):
    type: str
    requirements: List[str]
    preferences: List[str]

class CommunicationPlan(TypedDict):
    stakeholders: List[str]
    communication_channels: List[str]
    timeline: List[Dict[str, str]]
    message_templates: Dict[str, str]


class EventReports(TypedDict):
    estimated_budget: Dict[str, float]
    map_data: MapData
    destination_report: str
    supplier_requirements: List[SupplierRequirement]
    risk_assessment: List[Dict[str, Any]]
    sustainability_metrics: Dict[str, Any]
    accessibility_checklist: List[str]
    local_regulations: Dict[str, Any]
    communication_plan: CommunicationPlan

class SupplierState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    requirements: Optional[Requirements]
    draft_timeline: Optional[List[DayTimeline]]
    approved_timeline: Optional[Dict[str, Any]]  # Now includes both timeline and additional_gathered_preferences
    final_draft: Optional[Dict[str, Any]]
    final_draft_with_suppliers: Optional[FinalDraftWithSuppliers]
    event_reports: Optional[EventReports]
    formatted_requirements: Optional[str]
    formatted_timeline: Optional[str]
    formatted_final_draft: Optional[str]
    formatted_final_draft_with_suppliers: Optional[str]
    formatted_event_reports: Optional[str]
    suppliers_found: bool
    additional_preferences: Optional[List[str]]

# Define the config schema
class GraphConfig(TypedDict):
    thread_id: str