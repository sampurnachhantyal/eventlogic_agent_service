# supplier_suggestion_agent/utils/generate_event_reports.py

import json
import asyncio
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from state import SupplierState, EventReports, FinalEventDraft
from draft_timeline_node import generate_llm_response

# Configuration
DEFAULT_CONFIG = {
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_retries": 3,
    "cache_size": 128,
}

async def generate_event_reports(state: SupplierState, config: Dict[str, Any]) -> Dict[str, Any]:
    requirements = state.get("requirements", {})
    timeline = state.get("draft_event_plan", {}).get("timeline", [])
    
    # Merge default config with provided config
    merged_config = {**DEFAULT_CONFIG, **config}
    
    llm = ChatOpenAI(temperature=merged_config["temperature"], model_name=merged_config["model_name"])
    
    # Create tasks for concurrent execution
    tasks = [
        generate_estimated_budget(llm, requirements, timeline),
        generate_map_data(llm, requirements, timeline),
        generate_destination_report(llm, requirements),
        generate_supplier_requirements(llm, requirements, timeline),
        generate_risk_assessment(llm, requirements, timeline),
        generate_sustainability_metrics(llm, requirements, timeline),
        generate_accessibility_checklist(llm, requirements),
        generate_local_regulations_compliance(llm, requirements),
        generate_communication_plan(llm, requirements, timeline)
    ]
    
    # Run tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Extract content from results and update DraftEventPlan
    draft_event: DraftEventPlan = state.get("draft_event_plan", DraftEventPlan())
    draft_event.update({
        "estimated_budget": results[0],
        "map_data": results[1],
        "destination_report": results[2],
        "supplier_requirements": results[3],
        "risk_assessment": results[4],
        "sustainability_metrics": results[5],
        "accessibility_checklist": results[6],
        "local_regulations": results[7],
        "communication_plan": results[8]
    })
    
    # Return the updated state with the complete draft event plan
    return {
        "draft_event_plan": draft_event,
        "messages": [AIMessage(content="I've generated all the reports for your event. Please review the complete draft event plan.")]
    }

async def generate_estimated_budget(llm: ChatOpenAI, requirements: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, float]:
    prompt = ChatPromptTemplate.from_template(
        """Based on the following event requirements and timeline, generate a detailed estimated budget:
        
        Event Requirements: {requirements}
        Event Timeline: {timeline}
        
        Please provide a breakdown of estimated costs for various categories such as venue, catering, accommodation, transportation, etc. 
        The output should be a JSON object with category names as keys and estimated costs as values. 
        Include a "total" key with the sum of all expenses.
        
        Example format:
        {{
            "venue": 5000,
            "catering": 3000,
            "accommodation": 7500,
            "transportation": 2000,
            "miscellaneous": 1500,
            "total": 19000
        }}
        """
    )
    
    return await generate_llm_response(llm, prompt, {"requirements": requirements, "timeline": timeline})


async def generate_map_data(llm: ChatOpenAI, requirements: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """Given the following event requirements and timeline, generate map data for the event location:
        
        Event Requirements: {requirements}
        Event Timeline: {timeline}
        
        Please provide the following information:
        1. Latitude and longitude of the event location (use approximate coordinates if exact location is unknown)
        2. A suitable radius (in meters) to encompass the event area
        3. Any known points of interest related to the event (e.g., venue, recommended hotels, attractions)
        
        The output should be a JSON object with the following structure:
        {{
            "center": {{"lat": 0.0, "lng": 0.0}},
            "radius": 1000,
            "points_of_interest": [
                {{"name": "Main Venue", "lat": 0.0, "lng": 0.0}},
                {{"name": "Recommended Hotel", "lat": 0.0, "lng": 0.0}}
            ]
        }}
        """
    )
    
    return await generate_llm_response(llm, prompt, {"requirements": requirements, "timeline": timeline})

async def generate_destination_report(llm: ChatOpenAI, requirements: Dict[str, Any]) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Create a comprehensive destination report for the following event:
        
        Event Requirements: {requirements}
        
        Your report should include:
        1. Overview of the location
        2. Climate and weather considerations
        3. Transportation options
        4. Local attractions and points of interest
        5. Cultural considerations and etiquette
        6. Food and dining recommendations
        7. Safety and emergency information
        
        Please format your response as a Markdown document with appropriate headings and bullet points.
        """
    )
    
    return await generate_llm_response(llm, prompt, requirements, None)

async def generate_supplier_requirements(llm: ChatOpenAI, requirements: Dict[str, Any], timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = ChatPromptTemplate.from_template(
        """Based on the following event requirements and timeline, generate a list of supplier requirements:
        
        Event Requirements: {requirements}
        Event Timeline: {timeline}
        
        For each type of supplier needed (e.g., venue, catering, AV equipment), provide:
        1. The type of supplier
        2. Specific requirements or specifications
        3. Any preferences or constraints
        
        The output should be a JSON array of objects, where each object represents a supplier type and its requirements.
        
        Example format:
        [
            {{
                "type": "Venue",
                "requirements": [
                    "Capacity for 100 people",
                    "AV equipment included",
                    "Accessible for wheelchair users"
                ],
                "preferences": [
                    "Natural lighting",
                    "Close to public transportation"
                ]
            }},
            {{
                "type": "Catering",
                "requirements": [
                    "Lunch and coffee breaks for 100 people",
                    "Vegetarian and vegan options available"
                ],
                "preferences": [
                    "Local, sustainable ingredients",
                    "Ability to accommodate food allergies"
                ]
            }}
        ]
        """
    )
    
    return await generate_llm_response(llm, prompt, {"requirements": requirements, "timeline": timeline})


async def generate_risk_assessment(llm: ChatOpenAI, requirements: Dict[str, Any], timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = ChatPromptTemplate.from_template(
        """Conduct a risk assessment for the following event:
        
        Event Requirements: {requirements}
        Event Timeline: {timeline}
        
        Please identify potential risks based on the event type, location, activities, and timeline. For each risk:
        1. Describe the risk
        2. Assess its likelihood (Low, Medium, High)
        3. Assess its potential impact (Low, Medium, High)
        4. Suggest mitigation strategies
        
        The output should be a JSON array of risk objects.
        
        Example format:
        [
            {{
                "risk": "Adverse weather affecting outdoor activities",
                "likelihood": "Medium",
                "impact": "High",
                "mitigation": [
                    "Have indoor backup venues",
                    "Provide weather-appropriate gear to participants",
                    "Monitor weather forecasts and communicate changes"
                ]
            }},
            {{
                "risk": "Food-related illness outbreak",
                "likelihood": "Low",
                "impact": "High",
                "mitigation": [
                    "Use reputable catering services",
                    "Ensure proper food handling and storage",
                    "Have medical staff on standby"
                ]
            }}
        ]
        """
    )
    
    return await generate_llm_response(llm, prompt, {"requirements": requirements, "timeline": timeline})


async def generate_sustainability_metrics(llm: ChatOpenAI, requirements: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """Calculate sustainability metrics and provide suggestions for reducing environmental impact for the following event:
        
        Event Requirements: {requirements}
        Event Timeline: {timeline}
        
        Please provide:
        1. Estimated carbon footprint (in CO2 equivalent)
        2. Estimated water usage
        3. Estimated waste generation
        4. Suggestions for reducing environmental impact
        
        The output should be a JSON object with metrics and an array of suggestions.
        
        Example format:
        {{
            "carbon_footprint": {{"value": 5000, "unit": "kg CO2e"}},
            "water_usage": {{"value": 10000, "unit": "liters"}},
            "waste_generation": {{"value": 500, "unit": "kg"}},
            "reduction_suggestions": [
                "Use virtual attendance options to reduce travel emissions",
                "Implement a composting program for food waste",
                "Use reusable or compostable dishware and utensils",
                "Choose a venue with renewable energy sources"
            ]
        }}
        """
    )
    
    return await generate_llm_response(llm, prompt, {"requirements": requirements, "timeline": timeline})

async def generate_accessibility_checklist(llm: ChatOpenAI, requirements: Dict[str, Any]) -> List[str]:
    prompt = ChatPromptTemplate.from_template(
        """Generate an accessibility checklist for the following event:
        
        Event Requirements: {requirements}
        
        Please provide a comprehensive list of accessibility requirements and considerations based on the venue and activities. 
        Include items related to physical accessibility, sensory accessibility, and cognitive accessibility.
        
        The output should be a JSON array of checklist items.
        
        Example format:
        [
            "Ensure all venues have wheelchair-accessible entrances and restrooms",
            "Provide sign language interpreters for all main sessions",
            "Offer alternative formats for printed materials (e.g., large print, Braille)",
            "Ensure adequate lighting in all areas",
            "Provide quiet spaces for attendees who may need breaks from sensory stimulation"
        ]
        """
    )
    
    return await generate_llm_response(llm, prompt, requirements, None)

async def generate_local_regulations_compliance(llm: ChatOpenAI, requirements: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """Based on the following event requirements, provide information on relevant local regulations and required permits:
        
        Event Requirements: {requirements}
        
        Please include:
        1. Necessary permits or licenses
        2. Relevant local laws or regulations
        3. Compliance requirements
        4. Suggested actions to ensure compliance
        
        The output should be a JSON object with categories for permits, regulations, and compliance actions.
        
        Example format:
        {{
            "permits_required": [
                "Special Event Permit",
                "Temporary Food Service Permit",
                "Noise Variance Permit"
            ],
            "relevant_regulations": [
                "Local noise ordinances restrict outdoor events after 10 PM",
                "Food handlers must have valid food safety certifications",
                "Venues must comply with fire safety codes and occupancy limits"
            ],
            "compliance_actions": [
                "Apply for Special Event Permit at least 30 days before the event",
                "Ensure all food vendors have proper certifications",
                "Schedule a fire marshal inspection of the venue",
                "Distribute noise ordinance information to all staff and vendors"
            ]
        }}
        """
    )
    
    return await generate_llm_response(llm, prompt, requirements, None)

async def generate_communication_plan(llm: ChatOpenAI, requirements: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """Create a communication plan template for the following event:
        
        Event Requirements: {requirements}
        Event Timeline: {timeline}
        
        Please provide:
        1. A list of key stakeholders
        2. Communication channels to be used
        3. A timeline of key communications (aligned with the event timeline)
        4. Templates for important messages (e.g., save the date, registration information, post-event survey)
        
        The output should be a JSON object with sections for stakeholders, channels, timeline, and message templates.
        
        Example format:
        {{
            "stakeholders": [
                "Attendees",
                "Speakers",
                "Sponsors",
                "Venue Staff",
                "Volunteers"
            ],
            "communication_channels": [
                "Email",
                "Event Website",
                "Social Media",
                "Mobile App"
            ],
            "timeline": [
                {{"date": "2024-06-01", "action": "Send save the date to all stakeholders"}},
                {{"date": "2024-07-15", "action": "Open registration and send announcement"}},
                {{"date": "2024-08-30", "action": "Send final event details to attendees"}}
            ],
            "message_templates": {{
                "save_the_date": "Save the Date for [Event Name] on [Event Date]! Join us for [brief description]. More details coming soon.",
                "registration_open": "Registration is now open for [Event Name]! Early bird pricing available until [date]. Register now at [link].",
                "post_event_survey": "Thank you for attending [Event Name]! We value your feedback. Please take a moment to complete our survey: [link]"
            }}
        }}
        """
    )
    
    return await generate_llm_response(llm, prompt, {"requirements": requirements, "timeline": timeline})

# Make sure to update their signatures to include the timeline parameter where relevant