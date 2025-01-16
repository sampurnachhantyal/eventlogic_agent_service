# supplier_suggestion_agent/utils/tools_config.py

from ..database import get_db_connection, ensure_connection
from typing import Dict, List, Any
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# Define the base URL for the API
BASE_URL = os.environ.get('API_BASE_URL', 'http://host.docker.internal:8000')
# For local development, use 'http://localhost:8000'
# For Langgraph studio and docker use 'http://host.docker.internal:8000'


def fetch_supplier_categories() -> Dict[str, Dict[str, any]]:
    logger.debug("Fetching supplier categories from database")
    connection = get_db_connection()
    ensure_connection()
    categories = {}
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, name, en_name, parent_category_id, type
                FROM category
                WHERE category_group = 'SUPPLIER' AND deleted = false
            """)
            rows = cursor.fetchall()
            logger.debug(f"Fetched {len(rows)} categories from database")
            for row in rows:
                id, name, en_name, parent_category_id, type = row
                categories[id] = {
                    'name': name,
                    'en_name': en_name,
                    'parent_category_id': parent_category_id,
                    'type': type
                }
                logger.debug(f"Added category: id={id}, name={name}, en_name={en_name}")
    except Exception as e:
        logger.error(f"Error fetching supplier categories: {e}")
    finally:
        connection.close()
    
    logger.debug(f"Total categories fetched: {len(categories)}")
    return categories

SUPPLIER_CATEGORIES = fetch_supplier_categories()
logger.debug(f"SUPPLIER_CATEGORIES: {SUPPLIER_CATEGORIES}")


# List of allowed category names (in English)
ALLOWED_CATEGORIES = [
    "activity",
    "hotel",
    "conference_meeting_space",
    "event_space",
    "party_space",
    "restaurant",
    "bus",
    "Transportation",
    "Catering",
    "Yoga",
]

# Create a filtered CATEGORY_MAP with only the allowed categories
CATEGORY_MAP = {
    category['en_name']: category['name']
    for category in SUPPLIER_CATEGORIES.values()
    if category['en_name'] in ALLOWED_CATEGORIES
}

REVERSE_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}

# Define the function to get category IDs from category names
# I believe we might need to update this function to handle the mapping of getting the correct category names based on the content and use an llm for that.. Or implement it in the final draft node
def get_category_ids(category_name: str) -> List[int]:
    logger.debug(f"Getting category IDs for: {category_name}")
    category_ids = []
    for id, category in SUPPLIER_CATEGORIES.items():
        db_name = category['name'].lower()
        db_en_name = category['en_name'].lower() if category['en_name'] else ''
        search_name = category_name.lower().rstrip('s')  # Remove trailing 's' to match both singular and plural

        if search_name in db_name or search_name in db_en_name:
            logger.debug(f"Name match found for category: {category}")
            if any(allowed.lower().startswith(db_en_name) for allowed in ALLOWED_CATEGORIES):
                category_ids.append(id)
                logger.debug(f"Added category ID: {id}")
            else:
                logger.debug(f"Category {category['en_name']} not in ALLOWED_CATEGORIES")
        else:
            logger.debug(f"No match for category: {category['name']} / {category['en_name']}")
    logger.debug(f"Returning category IDs: {category_ids}")
    return category_ids

# Valid sort fields for the supplier search
VALID_SORT_FIELDS = ['supplier_name', 'rating', 'created_at', 'checked_at', 'created_in_el']


# Update EVENT_DATA_MAPPING
EVENT_DATA_MAPPING = {
    "id": "event.id",
    "event_type": "event.name",
    "start_date": "event.fromDate",
    "end_date": "event.toDate",
    "participants": "event.participantAmount",
    "location": "event.eventAddress.displayAddress",
    "additional_requirements": "event.extraRequirements",
    "event_contents": "event.requests",
    "content": "name",
    "offers": "requestOffers",
    "parts": "offerParts",
    "name": "name",
    "amount": "amount",
    "amount_type": "amountType.name",
    "start_time": "dateTimeFrom",
    "end_time": "dateTimeTo"
}

# Reverse mapping for updating events
REVERSE_EVENT_DATA_MAPPING = {
    "id": ["event", "id"],
    "event_type": ["event", "name"],
    "start_date": ["event", "fromDate"],
    "end_date": ["event", "toDate"],
    "participants": ["event", "participantAmount"],
    "location": ["event", "eventAddress", "displayAddress"],
    "additional_requirements": ["event", "extraRequirements"],
    "event_contents": ["event", "requests"],
    "content": ["request", "name"],
    "offers": ["request", "requestOffers"],
    "parts": ["offer", "offerParts"],
    "name": ["part", "name"],
    "amount": ["part", "amount"],
    "amount_type": ["part", "amountType", "name"],
    "start_time": ["part", "dateTimeFrom"],
    "end_time": ["part", "dateTimeTo"]
}

def map_event_data(data: Any, mapping: Dict[str, Any]) -> Any:
    """
    Recursively map data from one structure to another based on the provided mapping.
    """
    if isinstance(mapping, str):
        keys = mapping.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value
    if isinstance(mapping, dict):
        return {k: map_event_data(data, v) for k, v in mapping.items()}
    return data

def reverse_map_event_data(data, reverse_mapping):
    """
    Recursively map data from the simplified structure back to the original structure.
    """
    result = {}
    for key, value in data.items():
        if key in reverse_mapping:
            try:
                target = result
                for path_part in reverse_mapping[key][:-1]:
                    target = target.setdefault(path_part, {})
                target[reverse_mapping[key][-1]] = value
            except Exception as e:
                logger.error(f"Error mapping key '{key}': {str(e)}")
        else:
            logger.warning(f"No reverse mapping found for key: {key}")
    return result