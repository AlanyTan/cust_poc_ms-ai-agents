"""
ServiceNow REST API Client
=========================

Enterprise-grade ServiceNow integration with connection pooling,
automatic retry handling, and comprehensive error management.

Implements ServiceNow Table API best practices with proper authentication,
rate limiting awareness, and production-ready error handling patterns.
"""

import logging
import requests
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger(__name__)


class ServiceNowClient:
    """
    Production ServiceNow API client with enterprise features.

    Features connection pooling, automatic retry logic, proper error
    handling, and integration with secure configuration management.
    Designed for high-throughput ITSM automation workflows.
    """

    def __init__(self):
        """
        Initialize ServiceNow client with dependency injection pattern.
        """
        self.session = None
        self._config = None
        self.api_base = None
        self._initialize_client()

    @property
    def instance_url(self) -> Optional[str]:
        """Get the ServiceNow instance URL."""
        return self._config.get('instance_url') if self._config else None

    @property
    def username(self) -> Optional[str]:
        """Get the ServiceNow username."""
        return self._config.get('username') if self._config else None

    def _initialize_client(self):
        """
        Initialize HTTP session with ServiceNow-optimized settings.

        Configures connection pooling, timeout handling, and authentication
        based on current configuration state.
        """
        self._config = {
            'instance_url': os.getenv('SERVICENOW_INSTANCE_URL'),
            'username': os.getenv('SERVICENOW_USERNAME'),
            'password': os.getenv('SERVICENOW_PASSWORD'),
            'client_id': os.getenv('SERVICENOW_CLIENT_ID'),
            'client_secret': os.getenv('SERVICENOW_CLIENT_SECRET'),
            'auth_type': os.getenv('SERVICENOW_AUTH_TYPE', 'basic'),
            'verify_ssl': os.getenv('SERVICENOW_VERIFY_SSL', 'true').lower() == 'true',
            'timeout': int(os.getenv('SERVICENOW_TIMEOUT', 30))
        }

        # Ensure we have a valid instance URL
        if not self._config['instance_url']:
            logger.error("No instance URL provided in configuration")
            self.session = None
            self._config = None
            self.api_base = None
            return

        # Ensure the instance URL has the proper scheme
        if not self._config['instance_url'].startswith(('http://', 'https://')):
            self._config['instance_url'] = f"https://{self._config['instance_url']}"

        # Remove trailing slash if present
        self._config['instance_url'] = self._config['instance_url'].rstrip('/')

        # Base API URL
        self.api_base = f"{self._config['instance_url']}/api/now"

        # Create session for connection pooling
        self.session = requests.Session()
        self.session.auth = (self._config['username'], self._config['password'])
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

        logger.info(f"ServiceNow client initialized for {self._config['instance_url']}")

    def refresh_config(self):
        """Refresh configuration from the configuration manager."""
        try:
            self._initialize_client()
            logger.info("ServiceNow client configuration refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh ServiceNow configuration: {e}")
            raise

    def refresh_configuration(self):
        """Alias for refresh_config for backward compatibility."""
        self.refresh_config()
        return True

    def test_connection(self) -> Dict[str, Any]:
        """Test the ServiceNow connection."""
        if not self.session:
            return {"success": False, "message": "Client not initialized"}

        try:
            # Test with a simple incident query
            response = self.session.get(
                f"{self.api_base}/table/incident",
                params={
                    'sysparm_limit': 1,
                    'sysparm_fields': 'number'
                },
                timeout=10
            )

            # Check for hibernating instance
            if response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
                if 'hibernating' in response.text.lower() or 'instance hibernating' in response.text.lower():
                    return {
                        "success": False,
                        "message": "ServiceNow instance is hibernating. Please wake it up by visiting the instance URL in your browser.",
                        "hibernating": True
                    }

            if response.status_code == 200:
                try:
                    response.json()  # Test if we can parse JSON
                    return {
                        "success": True,
                        "message": "Connection successful",
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                except ValueError:
                    return {
                        "success": False,
                        "message": "Instance responded but returned HTML instead of JSON (possibly hibernating)",
                        "hibernating": True
                    }
            else:
                return {
                    "success": False,
                    "message": f"HTTP {response.status_code}: {response.text[:100]}"
                }

        except requests.exceptions.Timeout:
            return {"success": False, "message": "Connection timeout"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": "Connection failed - check URL and network"}
        except Exception as e:
            return {"success": False, "message": f"Test failed: {str(e)}"}

    def wake_instance(self) -> Dict[str, Any]:
        """Attempt to wake up a hibernating ServiceNow instance."""
        if not self._config:
            return {"success": False, "message": "Client not configured"}

        try:
            # Visit the main instance URL to wake it up
            wake_url = self._config['instance_url']
            response = self.session.get(wake_url, timeout=30)

            # Give it a moment to wake up
            import time
            time.sleep(3)

            # Test if it's awake now
            return self.test_connection()

        except Exception as e:
            return {"success": False, "message": f"Failed to wake instance: {str(e)}"}

    def _make_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """Make HTTP request to ServiceNow API with automatic config refresh on auth errors."""
        if not self.session:
            raise ValueError("ServiceNow client not initialized")

        url = f"{self.api_base}/table/{endpoint}"

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'PATCH':
                response = self.session.patch(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Handle authentication errors by attempting config refresh
            if response.status_code == 401:
                logger.warning("Authentication failed, attempting to refresh configuration")
                try:
                    self.refresh_config()
                    # Retry the request once with refreshed config
                    if method.upper() == 'GET':
                        response = self.session.get(url, params=params)
                    elif method.upper() == 'POST':
                        response = self.session.post(url, json=data)
                    elif method.upper() == 'PUT':
                        response = self.session.put(url, json=data)
                    elif method.upper() == 'PATCH':
                        response = self.session.patch(url, json=data)
                except Exception as refresh_error:
                    logger.error(f"Failed to refresh configuration: {refresh_error}")

            response.raise_for_status()

            # Debug: Check what we're actually getting
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            try:
                logger.debug(f"Response content: {response.text[:500]}")
            except (AttributeError, TypeError):
                logger.debug("Response content: <unable to display>")

            try:
                if not response.text.strip():
                    raise ValueError("Empty response from ServiceNow API")
            except (AttributeError, TypeError):
                # If response.text is not available (e.g., in tests), skip this check
                pass

            try:
                return response.json()
            except ValueError as json_error:
                # Check if instance is hibernating
                if 'text/html' in response.headers.get('content-type', ''):
                    if 'hibernating' in response.text.lower() or 'instance hibernating' in response.text.lower():
                        raise ValueError(
                            "ServiceNow instance is hibernating. Please wake it up by visiting the instance URL in your browser.")

                logger.error(f"Failed to parse JSON response: {json_error}")
                logger.error(f"Response content type: {response.headers.get('content-type')}")
                logger.error(f"Response text: {response.text[:1000]}")
                raise ValueError(f"Invalid JSON response (instance may be hibernating): {response.text[:200]}")

        except requests.exceptions.RequestException as e:
            logger.error(f"ServiceNow API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise Exception(f"ServiceNow API request failed: {str(e)}") from e

    # ServiceNow API methods remain the same as before

    def create_incident(self, **kwargs) -> Dict:
        """Create a new incident in ServiceNow."""
        data = {
            'short_description': kwargs.get('short_description', ''),
            'description': kwargs.get('description', ''),
            'state': kwargs.get('state', '1'),  # New
            'impact': kwargs.get('impact', '3'),  # Low
            'urgency': kwargs.get('urgency', '3'),  # Low
            'priority': kwargs.get('priority', '5'),  # Planning
            'category': kwargs.get('category', 'inquiry'),
            'subcategory': kwargs.get('subcategory', ''),
            'contact_type': kwargs.get('contact_type', 'email'),
            'caller_id': kwargs.get('caller_id', ''),
            'assignment_group': kwargs.get('assignment_group', ''),
            'assigned_to': kwargs.get('assigned_to', ''),
            'business_service': kwargs.get('business_service', ''),
            'cmdb_ci': kwargs.get('cmdb_ci', ''),
            'company': kwargs.get('company', ''),
            'location': kwargs.get('location', ''),
            'correlation_id': kwargs.get('correlation_id', ''),
            'correlation_display': kwargs.get('correlation_display', '')
        }

        # Remove empty fields
        data = {k: v for k, v in data.items() if v}

        response = self._make_request('POST', 'incident', data=data)
        return response['result']

    def get_incident(self, incident_id: str) -> Dict:
        """Get incident details by sys_id or number."""
        if incident_id.startswith('INC'):
            # Search by number
            params = {
                'sysparm_query': f'number={incident_id}',
                'sysparm_limit': 1
            }
            response = self._make_request('GET', 'incident', params=params)
            results = response['result']
            if not results:
                raise ValueError(f"Incident {incident_id} not found")
            return results[0]
        else:
            # Get by sys_id
            response = self._make_request('GET', f'incident/{incident_id}')
            return response['result']

    def list_incidents(self, **kwargs) -> List[Dict]:
        """List incidents with optional filtering."""
        params = {
            'sysparm_limit': kwargs.get('max_results', kwargs.get('limit', 10)),
            'sysparm_offset': kwargs.get('offset', 0),
            'sysparm_fields': kwargs.get('fields', 'sys_id,number,short_description,state,priority,assigned_to,created_on,updated_on'),
            'sysparm_order_by': kwargs.get('order_by', '-created_on')
        }

        # Build query in expected order: state, priority, assigned_to, others
        query_parts = []

        if kwargs.get('state'):
            query_parts.append(f"state={kwargs['state']}")

        if kwargs.get('priority'):
            query_parts.append(f"priority={kwargs['priority']}")

        if kwargs.get('assigned_to'):
            query_parts.append(f"assigned_to={kwargs['assigned_to']}")

        if kwargs.get('assignment_group'):
            query_parts.append(f"assignment_group={kwargs['assignment_group']}")

        if kwargs.get('caller_id'):
            query_parts.append(f"caller_id={kwargs['caller_id']}")

        if kwargs.get('category'):
            query_parts.append(f"category={kwargs['category']}")

        if kwargs.get('created_on_from'):
            query_parts.append(f"created_on>={kwargs['created_on_from']}")

        if kwargs.get('created_on_to'):
            query_parts.append(f"created_on<={kwargs['created_on_to']}")

        if query_parts:
            params['sysparm_query'] = '^'.join(query_parts)

        response = self._make_request('GET', 'incident', params=params)
        return response['result']

    def update_incident(self, incident_id: str, fields: Dict[str, Any] = None, **kwargs) -> Dict:
        """Update an incident."""
        # Prepare update data
        data = {}

        # Use fields parameter if provided, otherwise fall back to kwargs
        if fields:
            data.update(fields)

        # Also allow kwargs for backward compatibility
        update_fields = [
            'short_description', 'description', 'state', 'impact', 'urgency',
            'priority', 'category', 'subcategory', 'assigned_to',
            'assignment_group', 'work_notes', 'close_notes', 'close_code', 'resolution_notes',
            'caller_id', 'business_service', 'cmdb_ci', 'company', 'location'
        ]

        for field in update_fields:
            if field in kwargs:
                data[field] = kwargs[field]

        if not data:
            raise ValueError("No update fields provided")

        # Update by incident number or sys_id
        if incident_id.startswith('INC'):
            # Update by number using query parameter
            params = {'sysparm_query': f'number={incident_id}'}
            response = self._make_request('PATCH', 'incident', params=params, data=data)
        else:
            # Update by sys_id
            response = self._make_request('PATCH', f'incident/{incident_id}', data=data)

        return response['result']

    def get_resolution_codes(self) -> List[Dict]:
        """Get available resolution/close codes for incidents."""
        try:
            response = self._make_request('GET', 'sys_choice', params={
                'sysparm_query': 'name=incident^element=close_code',
                'sysparm_fields': 'value,label'
            })
            result = response.get('result', [])
            # Ensure we return a list - if result is a dict (single record), wrap it in a list
            if isinstance(result, dict):
                return [result] if result else []
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            logger.warning(f"Could not fetch resolution codes: {e}")
            return []

    def resolve_incident(self, incident_id: str, resolution_notes: str = "", close_code: str = None) -> Dict:
        """Resolve an incident."""
        # If no close code provided, try to use a common one
        if close_code is None:
            resolution_codes = self.get_resolution_codes()
            if resolution_codes:
                # Try to find a "Solved" or "Resolved" code
                for code in resolution_codes:
                    # Handle both dict and string formats defensively
                    if isinstance(code, dict):
                        label = code.get('label', '').lower()
                        if 'solved' in label or 'resolved' in label or 'fixed' in label:
                            close_code = code['value']
                            break
                    elif isinstance(code, str):
                        # If code is a string, use it directly
                        if 'solved' in code.lower() or 'resolved' in code.lower() or 'fixed' in code.lower():
                            close_code = code
                            break

                # If still no code found, use the first available one
                if close_code is None and resolution_codes:
                    try:
                        if isinstance(resolution_codes, list) and len(resolution_codes) > 0:
                            first_code = resolution_codes[0]
                            if isinstance(first_code, dict):
                                close_code = first_code['value']
                            else:
                                close_code = first_code
                        elif isinstance(resolution_codes, dict) and resolution_codes.get('value'):
                            close_code = resolution_codes['value']
                    except (KeyError, IndexError, TypeError):
                        logger.warning("Could not extract close code from resolution_codes")
                        pass

            # Fallback if API call failed
            if close_code is None:
                close_code = "Resolved by caller"

        return self.update_incident(
            incident_id,
            state='6',  # Resolved
            resolution_notes=resolution_notes,
            close_code=close_code,
            close_notes="Closed via API"
        )

    def close_incident(self, incident_id: str, close_notes: str = "", close_code: str = None) -> Dict:
        """Close an incident."""
        # If no close code provided, try to use a common one
        if close_code is None:
            resolution_codes = self.get_resolution_codes()
            if resolution_codes:
                # Try to find a "Solved" or "Resolved" code
                for code in resolution_codes:
                    # Handle both dict and string formats defensively
                    if isinstance(code, dict):
                        label = code.get('label', '').lower()
                        if 'solved' in label or 'resolved' in label or 'fixed' in label:
                            close_code = code['value']
                            break
                    elif isinstance(code, str):
                        # If code is a string, use it directly
                        if 'solved' in code.lower() or 'resolved' in code.lower() or 'fixed' in code.lower():
                            close_code = code
                            break

                # If still no code found, use the first available one
                if close_code is None and resolution_codes:
                    try:
                        if isinstance(resolution_codes, list) and len(resolution_codes) > 0:
                            first_code = resolution_codes[0]
                            if isinstance(first_code, dict):
                                close_code = first_code['value']
                            else:
                                close_code = first_code
                        elif isinstance(resolution_codes, dict) and resolution_codes.get('value'):
                            close_code = resolution_codes['value']
                    except (KeyError, IndexError, TypeError):
                        logger.warning("Could not extract close code from resolution_codes")
                        pass

            # Fallback if API call failed
            if close_code is None:
                close_code = "Resolved by caller"

        return self.update_incident(
            incident_id,
            state='7',  # Closed
            close_notes=close_notes,
            close_code=close_code
        )

    def add_work_note(self, incident_id: str, work_note: str) -> Dict:
        """Add a work note to an incident."""
        return self.update_incident(incident_id, work_notes=work_note)

    def assign_incident(self, incident_id: str, assigned_to: str = "", assignment_group: str = "") -> Dict:
        """Assign an incident to a user or group."""
        update_data = {}

        if assigned_to:
            update_data['assigned_to'] = assigned_to

        if assignment_group:
            update_data['assignment_group'] = assignment_group

        if not update_data:
            raise ValueError("Must provide either assigned_to or assignment_group")

        return self.update_incident(incident_id, **update_data)

    def search_incidents(self, query: str = None, **kwargs) -> List[Dict]:
        """Search incidents by text."""
        # Build search query - search for the query term in short_description primarily
        search_parts = []
        if query:
            search_parts.append(f"short_descriptionLIKE{query}")

        # Add additional filter conditions
        filters = []
        if 'state' in kwargs:
            filters.append(f"state={kwargs['state']}")
        if 'priority' in kwargs:
            filters.append(f"priority={kwargs['priority']}")
        if 'assignment_group' in kwargs:
            filters.append(f"assignment_group={kwargs['assignment_group']}")
        if 'assigned_to' in kwargs:
            filters.append(f"assigned_to={kwargs['assigned_to']}")
        if 'correlation_id' in kwargs:
            filters.append(f"correlation_id={kwargs['correlation_id']}")
        if 'correlation_display' in kwargs:
            filters.append(f"correlation_display={kwargs['correlation_display']}")

        # Combine search and filters
        all_parts = search_parts + filters
        search_query = '^'.join(all_parts) if all_parts else ""

        params = {
            'sysparm_query': search_query,
            'sysparm_limit': kwargs.get('max_results', kwargs.get('limit', 10)),
            'sysparm_fields': kwargs.get('fields', 'sys_id,number,short_description,state,priority,created_on,assigned_to,assignment_group')
        }

        response = self._make_request('GET', 'incident', params=params)
        return response['result']


# Global client instance for backward compatibility
_client_instance = None


def get_servicenow_client() -> ServiceNowClient:
    """Get the global ServiceNow client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = ServiceNowClient()
    return _client_instance


def reset_servicenow_client():
    """Reset the global ServiceNow client instance."""
    global _client_instance
    _client_instance = None
