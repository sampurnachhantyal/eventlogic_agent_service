
import logging
from typing import Dict, Any, List, Optional, Literal
from langchain_core.tools import tool
from datetime import datetime
from  ..database import get_db_connection, ensure_connection
from .tools_config import ALLOWED_CATEGORIES, BASE_URL, VALID_SORT_FIELDS, SUPPLIER_CATEGORIES, get_category_ids, EVENT_DATA_MAPPING, map_event_data # Importing the SUPPLIERS_CATEGORIES here so they should be loaded even if they are not used in the tool, as they are needed by get category ids
from .tools_utils import align_data_for_application, create_event_in_application, add_request_offer_part, add_supplier_to_request, add_or_update_event_content, add_supplier_to_request_and_send, prepare_event_contents, prepare_parts_data, prepare_supplier_data, get_event_request_ids, get_content_details
from ..utils.state import SupplierState
import requests
import time
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_community.tools import TavilySearchResults
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from dotenv import load_dotenv
import os

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Define the input schema for the tool
class CreateEventToolInput(BaseModel):
    user_email: str = Field(..., description="The email of the user creating the event")
    action: Literal["create", "create_and_add_suppliers", "create_add_suppliers_and_send_requests"] = Field(..., description="The action to perform")


class FetchSuppliersInput(BaseModel):
    name: Optional[str] = Field(None, description="Filter by supplier name")
    location: Optional[str] = Field(None, description="Filter by location (e.g., 'Stockholm', 'West coast of Sweden')")
    minRating: float = Field(0, description="Minimum rating (0-5)")
    maxRating: float = Field(5, description="Maximum rating (0-5)")
    categories: Optional[List[str]] = Field(None, description=f"Filter by categories. Allowed values: {', '.join(ALLOWED_CATEGORIES)}")
    limit: int = Field(10, description="Maximum number of results to return (1-10)")
    offset: int = Field(0, description="Number of results to skip (for pagination)")
    sort_by: str = Field('supplier_name', description=f"Field to sort by. Options: {', '.join(VALID_SORT_FIELDS)}")
    sort_order: Literal['asc', 'desc'] = Field('asc', description="Sort order ('asc' or 'desc')")

class GetEventDetailInput(BaseModel):
    event_id: int = Field(..., description="The ID of the event to fetch details for")

class AddOrUpdateEventContentInput(BaseModel):
    event_id: int = Field(..., description="The ID of the event to add or update content for")
    content_name: str = Field(..., description="The name of the content to add or update")
    content_id: Optional[int] = Field(None, description="The ID of the content to update (if updating existing content)")

class ContentPartInput(BaseModel):
    content_id: int = Field(..., description="The ID of the content to add a part to")
    name: str = Field(..., description="The name of the part")
    amount: int = Field(..., description="The amount for this part")
    amount_type: str = Field(..., description="The type of amount (e.g., 'PEOPLE' or 'PIECES')")
    timeless: bool = Field(..., description="Whether this part is timeless or not")
    date: Optional[str] = Field(None, description="The date for this part in 'YYYY-MM-DD' format (required if not timeless)")
    time: Optional[str] = Field(None, description="The start time for this part in 'HH:MM' format (required if not timeless)")
    duration_hours: Optional[str] = Field(None, description="The duration of this part in 'hours,minutes' format (e.g., '2,30' for 2 hours and 30 minutes, required if not timeless)")

class AddContentPartInput(BaseModel):
    part: ContentPartInput = Field(..., description="The part data to be added")

class AddSuppliersToContentInput(BaseModel):
    content_id: int = Field(..., description="The ID of the content to add suppliers to")
    supplier_ids: List[int] = Field(..., description="List of supplier IDs to add")
    send_requests: bool = Field(False, description="Whether to send requests to the suppliers immediately")

# Tool for creating an event in the EL application
@tool(parse_docstring=True)
def create_event_tool(
    user_email: str,
    action: Literal["create", "create_and_add_suppliers", "create_add_suppliers_and_send_requests"],
    state: Annotated[Dict[str, Any], InjectedState()]
) -> Dict[str, Any]:
    """
    Create an event in the EL application based on the final draft with suppliers from the state, user email, and specified action.

    Args:
        user_email: The email of the user creating the event.
        action: The action to perform. Options are "create", "create_and_add_suppliers", or "create_add_suppliers_and_send_requests".
    """
    final_draft_with_suppliers = state.get("final_draft_with_suppliers")
    if not final_draft_with_suppliers:
        return {"error": "final_draft_with_suppliers not found in state. Please build it first."}

    try:
        # Validate final_draft_with_suppliers structure
        if 'requirements' not in final_draft_with_suppliers or 'timeline' not in final_draft_with_suppliers:
            raise ValueError("Invalid final_draft_with_suppliers structure. Must contain 'requirements' and 'timeline'.")

        # Create event
        aligned_data = align_data_for_application(final_draft_with_suppliers, user_email)
        event_result = create_event_in_application(aligned_data)
        
        if not event_result or 'id' not in event_result:
            return {"error": "Failed to create event or retrieve event ID"}

        event_id = event_result['id']

        result = {"event": event_result, "contents": []}  # Initialize 'contents' as an empty list

        content_to_request_id = {}

        # Prepare event contents
        event_contents = prepare_event_contents(final_draft_with_suppliers)

        # Add content to the event and get request IDs sequentially with delay
        for content in event_contents:
            content_data = {
                "requests": [
                    {
                        "request": {
                            "name": content["name"]
                        }
                    }
                ]
            }
            add_content_result = add_or_update_event_content(event_id, content_data)
            if isinstance(add_content_result, dict) and add_content_result.get("status_code") == 204:
                # Content added successfully, now get the request ID
                content_details = get_content_details(event_id, content["name"])
                if content_details and content_details.get('offers'):
                    # Extract the ID from the first offer in the offers array
                    request_id = content_details['offers'][0].get('id')
                    if request_id:
                        content_to_request_id[content["name"]] = request_id
                        print(f"Added content: {content['name']} with request ID: {request_id}")
                    else:
                        return {"error": f"Failed to retrieve request ID for content: {content['name']}"}
                else:
                    return {"error": f"Failed to retrieve content details for: {content['name']}"}
            else:
                return {"error": f"Failed to add content: {content['name']}"}
            
            # Add a delay between requests (e.g., 1 second)
            time.sleep(0.5)

        # Add parts to each content
        for content in event_contents:
            request_id = content_to_request_id.get(content["name"])
            if not request_id:
                return {"error": f"Could not find request ID for content: {content['name']}"}
            
            parts_added = 0
            for part in content["parts"]:
                add_part_result = add_request_offer_part(request_id, part)
                if add_part_result:
                    parts_added += 1
                    print(f"Added part {part['name']} to content {content['name']}")
                else:
                    print(f"Failed to add part {part['name']} to content {content['name']}")
                
                # Add a delay between adding parts (e.g., 0.5 seconds)
                time.sleep(0.5)
            
            result["contents"].append({
                "name": content["name"],
                "request_id": request_id,
                "parts_added": parts_added
            })

        if action in ["create_and_add_suppliers", "create_add_suppliers_and_send_requests"]:
            send_requests = action == "create_add_suppliers_and_send_requests"
            suppliers_by_content = prepare_supplier_data(final_draft_with_suppliers["requirements"]["event_contents"], send_requests)
            for content_name, suppliers in suppliers_by_content.items():
                request_id = content_to_request_id.get(content_name)
                if not request_id:
                    return {"error": f"Could not find request ID for content: {content_name}"}
                
                supplier_data = {"suppliers": suppliers}
                if send_requests:
                    add_supplier_result = add_supplier_to_request_and_send(request_id, supplier_data)
                else:
                    add_supplier_result = add_supplier_to_request(request_id, supplier_data)
                
                if not add_supplier_result:
                    return {"error": f"Failed to add suppliers to content {content_name}"}
            
            result["added_suppliers"] = True
            if send_requests:
                result["sent_requests"] = True

        return result

    except requests.RequestException as e:
        logger.error(f"Network error in create_event_tool: {e}")
        return {"error": f"Network error: {str(e)}"}
    except ValueError as e:
        logger.error(f"Validation error in create_event_tool: {e}")
        return {"error": f"Validation error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in create_event_tool: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

    
@tool(args_schema=FetchSuppliersInput)
def fetch_suppliers_tool(
    name: Optional[str] = None,
    location: Optional[str] = None,
    minRating: float = 0,
    maxRating: float = 5,
    categories: Optional[List[str]] = None,
    matchStatus: str = None,
    limit: int = 10,
    offset: int = 0,
    sort_by: str = 'supplier_name',
    sort_order: Literal['asc', 'desc'] = 'asc',
    fields: str = 'id,supplier_name,town,country_code,status,match_status,el_supplier_id,description,rating,supplier_descriptions,reviews_summaries'
) -> Dict[str, Any]:
    """
    Fetch suppliers based on given criteria.

    Args:
        name: Filter by supplier name.
        location: Filter by location (e.g., "Stockholm", "West coast of Sweden").
        minRating: Minimum rating (0-5).
        maxRating: Maximum rating (0-5).
        categories: Filter by categories. Must be from the allowed list.
        limit: Maximum number of results to return (1-10).
        offset: Number of results to skip (for pagination).
        sort_by: Field to sort by. Must be one of the valid sort fields.
        sort_order: Sort order ('asc' or 'desc').
        fields: Comma-separated list of fields to fetch.

    Returns:
        Dict[str, Any]: A dictionary containing the total count and list of suppliers (max 10), or an error message.
    """
    debug_info = {
        #"input_params": locals(),
        #"steps": [],
        #"supplier_categories": SUPPLIER_CATEGORIES,
        #"allowed_categories": ALLOWED_CATEGORIES
    }

    try:
        limit = max(1, min(10, limit))
        #debug_info["steps"].append({"step": "Limit validation", "result": f"Limit set to {limit}"})

        if sort_by not in VALID_SORT_FIELDS:
            #debug_info["steps"].append({"step": "Sort field validation", "result": "Invalid sort field"})
            return {"error": f"Invalid sort_by field: {sort_by}. Allowed fields are: {', '.join(VALID_SORT_FIELDS)}", "debug_info": debug_info}

        category_ids = []
        invalid_categories = []
        if categories:
            #debug_info["steps"].append({"step": "Category processing", "categories": categories})
            for category in categories:
                category_step = {"category": category}
                if category in ALLOWED_CATEGORIES:
                    try:
                        ids = get_category_ids(category)
                        category_step["ids"] = ids
                        category_step["category_matching_process"] = []
                        for id, cat in SUPPLIER_CATEGORIES.items():
                            match_step = {
                                "id": id,
                                "name": cat['name'],
                                "en_name": cat['en_name'],
                                "name_match": cat['name'].lower() == category.lower(),
                                "en_name_match": cat['en_name'] and cat['en_name'].lower() == category.lower(),
                                "in_allowed_categories": cat['en_name'] in ALLOWED_CATEGORIES
                            }
                            category_step["category_matching_process"].append(match_step)
                        if ids:
                            category_ids.extend(ids)
                        else:
                            invalid_categories.append(category)
                            category_step["result"] = "No IDs found"
                    except Exception as e:
                        invalid_categories.append(category)
                        category_step["error"] = str(e)
                else:
                    invalid_categories.append(category)
                    category_step["result"] = "Not in ALLOWED_CATEGORIES"
                #debug_info["steps"].append(category_step)

            if not category_ids:
                error_message = f"No valid categories found. Invalid categories: {', '.join(invalid_categories)}. Allowed categories are: {', '.join(ALLOWED_CATEGORIES)}"
                #debug_info["steps"].append({"step": "Category validation", "result": "No valid categories"})
                return {"error": error_message, "debug_info": debug_info}

        #debug_info["steps"].append({"step": "Final category IDs", "ids": category_ids})

        # Get boundaries if location is provided
        boundaries = None
        if location:
            bounderies_url = f'{BASE_URL}/api/get_bounderies'
            bounderies_response = requests.get(bounderies_url, params={'location': location})
            bounderies_response.raise_for_status()
            bounderies_data = bounderies_response.json()
            boundaries = bounderies_data.get('boundaries')
            #debug_info["steps"].append({"step": "Get boundaries", "result": "Boundaries fetched" if boundaries else "No boundaries found"})

        # Prepare parameters for supplier fetch request
        request_url = f'{BASE_URL}/api/suppliers/agent_filter'
        params = {
            'fields': fields,
            'name': name,
            'minRating': minRating,
            'maxRating': maxRating,
            'matchStatus': matchStatus,
            'limit': limit,
            'offset': offset,
            'sort_by': sort_by,
            'sort_order': sort_order
        }

        if boundaries:
            params.update({
                'north': boundaries['north']['latitude'],
                'south': boundaries['south']['latitude'],
                'east': boundaries['east']['longitude'],
                'west': boundaries['west']['longitude']
            })

        if category_ids:
            params['categories'] = ','.join(map(str, category_ids))

        #debug_info["steps"].append({"step": "Final request parameters", "params": params})

        response = requests.get(request_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        #debug_info["steps"].append({"step": "API response", "total": data.get('total'), "suppliers_count": len(data.get('suppliers', []))})

        return {
            "total": data['total'],
            "suppliers": data['suppliers'][:10],  # Ensure we never return more than 10 suppliers
            #"debug_info": debug_info
        }
    
    except requests.RequestException as e:
        debug_info["steps"].append({"step": "API request", "error": str(e)})
        return {"error": f"Network error while fetching supplier list: {str(e)}. Please try again later.", "debug_info": debug_info}
    except Exception as e:
        debug_info["steps"].append({"step": "Unexpected error", "error": str(e)})
        return {"error": f"An unexpected error occurred: {str(e)}. Please try again later.", "debug_info": debug_info}

# Update the docstring with the actual values
fetch_suppliers_tool.__doc__ = fetch_suppliers_tool.__doc__.format(
    allowed_categories=', '.join(ALLOWED_CATEGORIES),
    valid_sort_fields=', '.join(VALID_SORT_FIELDS)
)


@tool
def add_supplier_to_db_tool(supplier_name: Optional[str] = None, location: Optional[str] = None, place_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Add a supplier to the Potential Suppliers database based on name, location, or place_id.
    
    Args:
        supplier_name (str, optional): Name of the supplier to add.
        location (str, optional): Country or town where the supplier is located.
        place_id (str, optional): Add supplier based on place_id.

    Returns:
        Dict[str, Any]: A dictionary containing the supplier data and relevant links, or an error message.
    """
    if not supplier_name and not place_id:
        return {"error": "Insufficient information provided. Please provide supplier name or place_id."}
    
    logger.debug(f"Attempting to add supplier: {supplier_name or place_id}")

    try:
        request_url = f'{BASE_URL}/api/add_single_supplier'
        data = {
            "supplier_name": supplier_name,
            "location": location,
            "place_id": place_id
        }

        response = requests.post(request_url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"Supplier search result: {result}")

        return result  # This now includes message, filtered_suppliers, and potential_suppliers

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error occurred while adding supplier: {e}")
        return {"error": f"Network error while adding supplier: {str(e)}. Please try again later."}


# Get event detail tool
@tool(args_schema=GetEventDetailInput)
def get_event_detail_tool(event_id: int) -> Dict[str, Any]:
    """
    Fetch event details from the EL application and return a mapped version using the data mapping.
    
    Args:
        event_id (int): The ID of the event to fetch details for.
    
    Returns:
        Dict[str, Any]: A dictionary containing the mapped event details, or an error message if the fetch failed.
    """
    try:
        result = requests.get(f"{BASE_URL}/api/get_event_detail/{event_id}")
        result.raise_for_status()
        event_details = result.json()
        
        if not isinstance(event_details, dict):
            raise ValueError(f"Unexpected response format: {type(event_details)}")
        
        mapped_details = map_event_data(event_details, EVENT_DATA_MAPPING)
        
        # Safely convert timestamps to date strings
        for date_field in ['start_date', 'end_date']:
            if date_field in mapped_details and mapped_details[date_field] is not None:
                try:
                    mapped_details[date_field] = datetime.fromtimestamp(mapped_details[date_field] / 1000).strftime('%Y-%m-%d')
                except (ValueError, TypeError, OverflowError):
                    logger.warning(f"Failed to convert {date_field} to date string")
        
        return mapped_details
    except requests.RequestException as e:
        logger.error(f"Network error fetching event details: {e}")
        return {"error": f"Failed to fetch event details due to network error: {str(e)}"}
    except (ValueError, AttributeError, TypeError) as e:
        logger.error(f"Error processing event details: {e}")
        return {"error": f"Failed to process event details: {str(e)}"}


@tool(args_schema=AddOrUpdateEventContentInput)
def add_or_update_event_content_tool(event_id: int, content_name: str, content_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Add new content or update existing content for an event in the EL application,
    then retrieve the latest content details.
    
    Args:
        event_id (int): The ID of the event to add or update content for.
        content_name (str): The name of the content to add or update.
        content_id (Optional[int]): The ID of the content to update (if updating existing content).
    
    Returns:
        Dict[str, Any]: A dictionary containing the result of the operation and the latest content details.
    """
    try:
        content_data = {
            "requests": [
                {
                    "request": {
                        "name": content_name,
                        "id": content_id
                    }
                }
            ]
        }
        
        logger.info(f"Updating content: {content_name} for event: {event_id}")
        response = add_or_update_event_content(event_id, content_data)
        logger.info(f"Update response: {response}")
        
        if isinstance(response, dict) and response.get('status_code') == 204:
            # Content was successfully added/updated
            logger.info("Content update successful, fetching latest details")
            latest_content_details = get_content_details(event_id, content_name)
            
            if latest_content_details:
                return {
                    "success": True,
                    "message": response.get('message', 'Content added/updated successfully'),
                    "status_code": response.get('status_code'),
                    "content_details": latest_content_details
                }
            else:
                logger.warning("Content updated but details could not be retrieved")
                return {
                    "success": True,
                    "message": "Content added/updated successfully, but details could not be retrieved",
                    "status_code": response.get('status_code')
                }
        else:
            logger.error(f"Unexpected response from content update: {response}")
            return {
                "success": False,
                "error": f"Failed to add or update content. Unexpected response: {response}",
                "status_code": response.get('status_code') if isinstance(response, dict) else None
            }
    
    except Exception as e:
        logger.error(f"Error adding or updating event content: {e}")
        return {
            "success": False,
            "error": f"Failed to add or update event content: {str(e)}",
            "status_code": None
        }


# Add content part to an existing content item
@tool
def add_content_part_tool(
    content_id: int,
    name: str,
    amount: int,
    amount_type: str,
    timeless: bool,
    date: Optional[str],
    time: Optional[str],
    duration_hours: Optional[str]
) -> Dict[str, Any]:
    """
    Add a new part to an existing content item in the EL application.

    Args:
        content_id: The ID of the content to add a part to
        name: The name of the part
        amount: The amount for this part
        amount_type: The type of amount (e.g., 'PEOPLE' or 'PIECES')
        timeless: Whether this part is timeless or not
        date: The date for this part in 'YYYY-MM-DD' format (required if not timeless)
        time: The start time for this part in 'HH:MM' format (required if not timeless)
        duration_hours: The duration of this part in 'hours,minutes' format (e.g., '2,30' for 2 hours and 30 minutes, required if not timeless)

    Returns:
        Dict[str, Any]: A dictionary containing the result of the operation.
    """
    try:
        # Validate input using PartData schema
        part_data = ContentPartInput(
            name=name,
            amount=amount,
            amount_type=amount_type,
            timeless=timeless,
            date=date,
            time=time,
            duration_hours=duration_hours,
        )
        # If part is not timeless, ensure all time-related fields are provided
        if not timeless and not all([date, time, duration_hours]):
            return {"success": False, "error": "Date, time, and duration_hours are required for non-timeless parts"}

        # Convert to dict for downstream processing
        prepared_part = part_data.model_dump()
        prepared_part["content_id"] = content_id
        # Use the prepared part to add to content
        result = add_request_offer_part(content_id, prepared_part)

        # Return success response
        if result:
            return {
                "success": True,
                "message": f"Part '{name}' added successfully to content {content_id}",
                "part_id": result.get("id"),
            }
        else:
            return {"success": False, "error": "Failed to add part"}

    except Exception as e:
        logger.error(f"Error adding content part: {e}")
        return {"success": False, "error": f"Failed to add content part: {str(e)}"}


    except Exception as e:
        logger.error(f"Error adding content part: {e}")
        return {"success": False, "error": f"Failed to add content part: {str(e)}"}


# Add suppliers to an existing content item
@tool(args_schema=AddSuppliersToContentInput)
def add_suppliers_to_content_tool(content_id: int, supplier_ids: List[int], send_requests: bool = False) -> Dict[str, Any]:
    """
    Add suppliers to an existing content item in the EL application and optionally send requests.
    
    Args:
        content_id (int): The ID of the content to add suppliers to.
        el_supplier_ids (List[int]): List of el_supplier IDs to add.
        send_requests (bool): Whether to send requests to the suppliers immediately.
    
    Returns:
        Dict[str, Any]: A dictionary containing the result of the operation.
    """
    try:
        supplier_data = {"suppliers": [{"id": supplier_id, "send": send_requests} for supplier_id in supplier_ids]}

        add_supplier_to_request(content_id, supplier_data)
        
        if send_requests:
            result = add_supplier_to_request_and_send(content_id, supplier_data)
        else:
            result = add_supplier_to_request(content_id, supplier_data)
        
        if result:
            return {"success": True, "message": f"Suppliers added successfully to content {content_id}", "sent_requests": send_requests}
        else:
            return {"success": False, "error": "Failed to add suppliers"}
    
    except Exception as e:
        logger.error(f"Error adding suppliers to content: {e}")
        return {"success": False, "error": f"Failed to add suppliers to content: {str(e)}"}


@tool
def fetch_temp_suppliers_tool(
    event_type: str,
    location: str,
    criteria: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch suppliers based on event_type, location, and optional criteria.
    """
    logger.debug(f"Fetching suppliers for event_type: {event_type}, location: {location}, criteria: {criteria}")
    try:
        connection = get_db_connection()
        ensure_connection()
        with connection.cursor() as cursor:
            # Fetch from potential_suppliers
            query = """
                SELECT * FROM potential_suppliers
                WHERE LOWER(event_type) = LOWER(%s)
                AND LOWER(location) = LOWER(%s)
            """
            params = [event_type, location]
            # Add criteria to the query if provided
            if criteria:
                for criterion in criteria:
                    query += " AND LOWER(criteria) LIKE LOWER(%s)"
                    params.append(f"%{criterion}%")
            cursor.execute(query, params)
            potential_suppliers = cursor.fetchall()

            # Fetch from filtered_apify_data where potential_supplier_id is NULL
            query = """
                SELECT * FROM filtered_apify_data
                WHERE potential_supplier_id IS NULL
                AND LOWER(event_type) = LOWER(%s)
                AND LOWER(location) = LOWER(%s)
            """
            params = [event_type, location]
            if criteria:
                for criterion in criteria:
                    query += " AND LOWER(criteria) LIKE LOWER(%s)"
                    params.append(f"%{criterion}%")
            cursor.execute(query, params)
            apify_suppliers = cursor.fetchall()

            # Combine suppliers
            suppliers = potential_suppliers + apify_suppliers

        # Convert suppliers to list of dicts
        supplier_list = []
        for supplier in suppliers:
            supplier_dict = dict(zip([desc[0] for desc in cursor.description], supplier))
            supplier_list.append(supplier_dict)

        return supplier_list
    except Exception as e:
        logger.error(f"Error fetching suppliers: {e}")
        return []


# Update the list of tools
tools = [
    TavilySearchResults(max_results=10),
    fetch_suppliers_tool,
    create_event_tool,
    add_supplier_to_db_tool,
    get_event_detail_tool,
    add_or_update_event_content_tool,
    add_content_part_tool,
    add_suppliers_to_content_tool,
]