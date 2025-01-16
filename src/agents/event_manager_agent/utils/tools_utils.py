# supplier_suggestion_agent/utils/tools_utils.py

import requests
from typing import Dict, Any, List
from dotenv import load_dotenv
import os
import time
from datetime import datetime
from datetime import datetime, timedelta
from agents.event_manager_agent.utils.tools_config import EVENT_DATA_MAPPING, map_event_data
import logging

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Define the base URL for the API
BASE_URL = os.environ.get('API_BASE_URL', 'http://host.docker.internal:8000')

def align_data_for_application(final_draft: Dict[str, Any], user_email: str) -> Dict[str, Any]:
    requirements = final_draft["requirements"]
    
    # Convert dates to datetime objects
    start_date = datetime.strptime(requirements["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(requirements["end_date"], "%Y-%m-%d")
    
    # Calculate event duration
    duration = (end_date - start_date).days + 1
    
    # Determine template ID based on duration
    if duration == 1:
        template_id = 1240021  # One day
    elif duration == 2:
        template_id = 1239948  # Two days
    elif duration == 3:
        template_id = 1227509  # Three days
    else:
        template_id = 1240021  # Default to one day if duration is unexpected
    
    # Convert dates to millisecond timestamps
    from_date = int(time.mktime(start_date.timetuple()) * 1000)
    to_date = int(time.mktime(end_date.timetuple()) * 1000)
    
    aligned_data = {
        "templateId": template_id,
        "fromDate": from_date,
        "toDate": to_date,
        "name": requirements["event_type"],
        "participantAmount": str(requirements["participants"]),
        "eventAddress": {
            "country": "Sweden",
            "displayAddress": requirements["location"]
        },
        "extraRequirements": requirements.get("additional_requirements", ""),
        "email": user_email,
        "escapeAccommodation": requirements.get("overnight_guests", "") == "",
        "eventCoach": False,
        "interestedInOtherDates": False,
        "isJulbordEvent": False,
        "isOnboardingEvent": False,
        "addAccommodationPartsNotPresentInTemplate": False
    }
    
    return aligned_data

def prepare_event_contents(final_draft: Dict[str, Any]) -> List[Dict[str, Any]]:
    requirements = final_draft["requirements"]
    
    event_contents = []
    for content in requirements.get("event_contents", []):
        aligned_content = {
            "name": content["content"],
            "parts": prepare_parts_data(content["parts"]),
            "potential_suppliers": content.get("potential_suppliers", [])
        }
        event_contents.append(aligned_content)
    
    return event_contents

def parse_duration(duration_str: str) -> int:
    """Parse a duration string and return the total minutes."""
    duration_parts = duration_str.split()
    total_minutes = 0
    for i in range(0, len(duration_parts), 2):
        value = float(duration_parts[i])
        unit = duration_parts[i+1].lower()
        if 'hour' in unit:
            total_minutes += int(value * 60)
        elif 'minute' in unit:
            total_minutes += int(value)
    return total_minutes

def prepare_parts_data(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parts_data = []
    
    for part in parts:
        part_data = {
            "name": part["name"],
            "timeless": part["timeless"],
            "amountType": {"name": part["amount_type"]},
            "amount": int(part["amount"]),
            "commentByCreator": ""
        }
        
        if not part["timeless"]:
            # Parse the date and time
            event_date = datetime.strptime(part["date"], "%Y-%m-%d")
            start_time = datetime.strptime(part["time"], "%H:%M")
            
            # Combine date and time
            start_datetime = event_date.replace(hour=start_time.hour, minute=start_time.minute)
            
            # Calculate end time
            duration_hours, duration_minutes = map(int, part["duration_hours"].split(','))
            duration = timedelta(hours=duration_hours, minutes=duration_minutes)
            end_datetime = start_datetime + duration
            
            # Convert to milliseconds since midnight
            start_time_ms = int((start_datetime - start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000)
            end_time_ms = int((end_datetime - end_datetime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000)
            
            # Convert the event date to milliseconds since epoch
            event_from_date_ms = int(event_date.timestamp() * 1000)
            
            part_data.update({
                "dateTimeFrom": start_time_ms,
                "dateTimeTo": end_time_ms,
                "eventFromDate": event_from_date_ms
            })
        
        parts_data.append(part_data)
    
    return parts_data


def prepare_supplier_data(event_contents: List[Dict[str, Any]], send: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    suppliers_by_content = {}
    for content in event_contents:
        content_name = content["name"]
        suppliers = [
            {"id": supplier["el_supplier_id"], "send": send} 
            for supplier in content.get("potential_suppliers", [])
            if supplier.get("el_supplier_id")
        ]
        if suppliers:
            suppliers_by_content[content_name] = suppliers
    return suppliers_by_content


# Helper function for the create_event_tool to get the request ids that have been added as these are not returned by the add or update content api
def get_event_request_ids(event_id: int) -> Dict[str, Any]:
    """
    Fetch event details from the EL application and extract request names and IDs.
    
    Args:
        event_id (int): The ID of the event to fetch details for.
    
    Returns:
        Dict[str, Any]: A dictionary mapping request names to their IDs, or an error message if the fetch failed.
    """
    try:
        result = requests.get(f"{BASE_URL}/api/get_event_detail/{event_id}")
        result.raise_for_status()
        event_details = result.json()
        
        if 'event' not in event_details or 'requests' not in event_details['event']:
            logger.error(f"Unexpected event details format: {event_details}")
            return {"error": "Unexpected event details format"}
        
        # Extract request names and IDs
        content_to_request_id = {}
        for request in event_details['event']['requests']:
            if 'name' in request and 'id' in request:
                content_to_request_id[request['name']] = request['id']
            else:
                logger.warning(f"Request missing name or id: {request}")
        
        if not content_to_request_id:
            logger.warning("No valid requests found in event details")
            return {"error": "No valid requests found in event details"}
        
        logger.info(f"Extracted content to request ID mapping: {content_to_request_id}")
        return content_to_request_id
    except requests.RequestException as e:
        logger.error(f"Error fetching event request IDs: {e}")
        return {"error": f"Failed to fetch event request IDs: {str(e)}"}


# Helper function used by the add or update content to get the content details back after adding or updating    
def get_content_details(event_id: int, content_name: str) -> Dict[str, Any]:
    """
    Fetch details for a specific content item within an event, using the EVENT_DATA_MAPPING.
    
    Args:
        event_id (int): The ID of the event.
        content_name (str): The name of the content to fetch details for.
    
    Returns:
        Dict[str, Any]: A dictionary containing the mapped content details, or None if not found.
    """
    try:
        logger.info(f"Fetching event details for event_id: {event_id}")
        result = requests.get(f"{BASE_URL}/api/get_event_detail/{event_id}")
        result.raise_for_status()
        event_details = result.json()
        
        if not isinstance(event_details, dict) or 'event' not in event_details or 'requests' not in event_details['event']:
            logger.error(f"Unexpected event details format: {event_details}")
            raise ValueError(f"Unexpected response format: {type(event_details)}")
        
        logger.info(f"Searching for content: {content_name}")
        for request in event_details['event']['requests']:
            if request['name'] == content_name:
                logger.info(f"Content found: {content_name}")
                # Map the content details according to EVENT_DATA_MAPPING
                mapped_content = {
                    "content": request.get('name'),
                    "offers": [],
                    "parts": []
                }
                
                for offer in request.get('requestOffers', []):
                    mapped_offer = {
                        "id": offer.get('request', {}).get('id'),
                        "status": offer.get('status', {}).get('name')
                    }
                    mapped_content['offers'].append(mapped_offer)
                    
                    for part in offer.get('offerParts', []):
                        mapped_part = {
                            "name": part.get('name'),
                            "amount": part.get('amount'),
                            "amount_type": part.get('amountType', {}).get('name'),
                            "start_time": part.get('dateTimeFrom'),
                            "end_time": part.get('dateTimeTo'),
                            "event_from_date": part.get('eventFromDate')
                        }
                        mapped_content['parts'].append(mapped_part)
                
                logger.info(f"Mapped content: {mapped_content}")
                return mapped_content
        
        logger.warning(f"Content not found: {content_name}")
        return None
    except requests.RequestException as e:
        logger.error(f"Error fetching event details: {e}")
        return None
    except (ValueError, AttributeError, TypeError) as e:
        logger.error(f"Error processing event details: {e}")
        return None


# Helper function to call the API
def call_api(endpoint: str, method: str = 'GET', data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generic function to call API endpoints"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == 'GET':
            response = requests.get(url, params=params)
        elif method == 'POST':
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"API call failed: {e}")
        raise

def create_event_in_application(event_data: Dict[str, Any]) -> Dict[str, Any]:
    return call_api('/api/create_event', method='POST', data=event_data)

def add_or_update_event_content(event_id: int, content_data: Dict[str, Any]) -> Dict[str, Any]:
    return call_api(f'/api/add_or_update_content/{event_id}', method='POST', data=content_data)

def add_request_offer_part(request_id: int, offer_part_data: Dict[str, Any]) -> Dict[str, Any]:
    return call_api(f'/api/add_request_offer_part/{request_id}', method='POST', data=offer_part_data)

def add_supplier_to_request(request_id: int, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
    return call_api(f'/api/add_supplier_to_request/{request_id}', method='POST', data=supplier_data)

def add_supplier_to_request_and_send(request_id: int, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
    return call_api(f'/api/add_supplier_to_request_and_send/{request_id}', method='POST', data=supplier_data)