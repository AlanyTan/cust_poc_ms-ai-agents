# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import secrets
from typing import Optional
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Body, FastAPI, Depends, HTTPException, status, Response
from opentelemetry import trace
import asyncio
import json
import os
from typing import AsyncGenerator, Optional, Dict

import fastapi
from fastapi import Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import logging
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    Agent,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    AsyncAgentEventHandler,
    RunStep
)
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentEvaluationRequest,
    AgentEvaluationSamplingConfiguration,
    AgentEvaluationRedactionConfiguration,
    EvaluatorIds
)

from .ms_resource_graph_client import get_tags, get_monitored_metrics

# Create a logger for this module
logger = logging.getLogger("azureaiapp")

# Set the log level for the azure HTTP logging policy to WARNING (or ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

tracer = trace.get_tracer(__name__)

# Define the directory for your templates.
directory = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=directory)

# Create a new FastAPI router
router = fastapi.APIRouter()

# Pydantic models for request/response documentation


class ChatRequest(BaseModel):
    message: str = Field(description="The user's message to send to the AI agent")

    class Config:
        schema_extra = {
            "example": {
                "message": "Hi, AI, please help me answer this question what is LLM?"
            }
        }


# Azure Monitor Alert Payload Models
class AlertConditionDimension(BaseModel):
    name: Optional[str] = Field(None, description="Name of the dimension")
    value: Optional[str] = Field(None, description="Value of the dimension")


class AlertConditionAllOf(BaseModel):
    metricName: str = Field(description="Name of the metric")
    metricNamespace: Optional[str] = Field(None, description="Namespace of the metric")
    operator: str = Field(description="Operator used in the condition (e.g., GreaterThan)")
    threshold: str = Field(description="Threshold value as string")
    timeAggregation: str = Field(description="Time aggregation method (e.g., Sum, Average)")
    dimensions: list[AlertConditionDimension] = Field(default=[], description="Metric dimensions")
    metricValue: float = Field(description="Current metric value")
    webTestName: Optional[str] = Field(None, description="Web test name for availability tests")


class AlertCondition(BaseModel):
    windowSize: str = Field(description="Time window size (ISO 8601 duration)")
    allOf: list[AlertConditionAllOf] = Field(description="List of conditions")
    windowStartTime: str = Field(description="Window start time (ISO 8601)")
    windowEndTime: str = Field(description="Window end time (ISO 8601)")


class AlertContext(BaseModel):
    properties: Optional[dict] = Field(None, description="Additional properties")
    conditionType: str = Field(description="Type of condition (e.g., WebtestLocationAvailabilityCriteria)")
    condition: AlertCondition = Field(description="Alert condition details")


class AlertEssentials(BaseModel):
    alertId: str = Field(description="Unique alert identifier")
    alertRule: str = Field(description="Name of the alert rule")
    severity: str = Field(description="Alert severity (e.g., Sev0, Sev1, Sev2, Sev3, Sev4)")
    signalType: str = Field(description="Type of signal (e.g., Metric, Log)")
    monitorCondition: str = Field(description="Monitor condition (e.g., Fired, Resolved)")
    monitoringService: str = Field(description="Monitoring service (e.g., Platform, ApplicationInsights)")
    alertTargetIDs: list[str] = Field(description="List of target resource IDs")
    configurationItems: list[str] = Field(description="Configuration items affected")
    originAlertId: str = Field(description="Original alert identifier")
    firedDateTime: str = Field(description="When the alert was fired (ISO 8601)")
    description: str = Field(description="Alert description")
    essentialsVersion: str = Field(description="Version of essentials schema")
    alertContextVersion: str = Field(description="Version of alert context schema")


class AlertData(BaseModel):
    essentials: AlertEssentials = Field(description="Essential alert information")
    alertContext: AlertContext = Field(description="Detailed alert context")
    customProperties: Optional[dict] = Field(None, description="Custom properties")


class AlertBody(BaseModel):
    schemaId: str = Field(description="Schema identifier for the alert")
    data: AlertData = Field(description="Alert data payload")


class MonitorPayload(BaseModel):
    headers: dict[str, str] = Field(description="HTTP headers from the monitor request")
    body: AlertBody = Field(description="Alert payload body")

    class Config:
        schema_extra = {
            "example": {
                "headers": {
                    "Content-Type": "application/json; charset=utf-8",
                    "Host": "prod-73.eastus.logic.azure.com",
                    "User-Agent": "IcMBroadcaster/1.0",
                    "X-CorrelationContext": "RkkKACgAAAACAAAAEAAOcr4CoiHuSZvSeZpqt4BRAQAQACSLkvJrA8RDhFCwUCrNEAQ="
                },
                "body": {
                    "schemaId": "azureMonitorCommonAlertSchema",
                    "data": {
                        "essentials": {
                            "alertId": "/subscriptions/11111111-1111-1111-1111-111111111111/providers/Microsoft.AlertsManagement/alerts/12345678-1234-1234-1234-1234567890ab",
                            "alertRule": "test-availabilityTest-test-applicationInsights",
                            "severity": "Sev4",
                            "signalType": "Metric",
                            "monitorCondition": "Fired",
                            "monitoringService": "Platform",
                            "alertTargetIDs": [
                                "/subscriptions/11111111-1111-1111-1111-111111111111/resourcegroups/test-RG/providers/microsoft.insights/components/test-applicationInsights"
                            ],
                            "configurationItems": ["test-applicationInsights"],
                            "originAlertId": "11111111-1111-1111-1111-111111111111_test-RG_microsoft.insights_metricalerts_test-availabilityTest-test-applicationInsights_1234567890",
                            "firedDateTime": "2025-08-29T18:00:16.117Z",
                            "description": "Alert rule description",
                            "essentialsVersion": "1.0",
                            "alertContextVersion": "1.0"
                        },
                        "alertContext": {
                            "properties": None,
                            "conditionType": "WebtestLocationAvailabilityCriteria",
                            "condition": {
                                "windowSize": "PT5M",
                                "allOf": [
                                    {
                                        "metricName": "Failed Location",
                                        "metricNamespace": None,
                                        "operator": "GreaterThan",
                                        "threshold": "2",
                                        "timeAggregation": "Sum",
                                        "dimensions": [],
                                        "metricValue": 5,
                                        "webTestName": "test-availabilityTest-test-applicationInsights"
                                    }
                                ],
                                "windowStartTime": "2025-08-29T18:00:16.117Z",
                                "windowEndTime": "2025-08-29T18:00:16.117Z"
                            }
                        },
                        "customProperties": {
                            "customKey1": "value1",
                            "customKey2": "value2"
                        }
                    }
                }
            }
        }


security = HTTPBasic()

username = os.getenv("WEB_APP_USERNAME")
password = os.getenv("WEB_APP_PASSWORD")
basic_auth = username and password

# Health check endpoint


@router.get("/healthz", summary="Health check endpoint")
async def healthz():
    return Response(status_code=200)


def authenticate(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> None:

    if not basic_auth:
        logger.info("Skipping authentication: WEB_APP_USERNAME or WEB_APP_PASSWORD not set.")
        return

    correct_username = secrets.compare_digest(credentials.username, username)
    correct_password = secrets.compare_digest(credentials.password, password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return


auth_dependency = Depends(authenticate) if basic_auth else None


def get_ai_project(request: Request) -> AIProjectClient:
    return request.app.state.ai_project


def get_agent_client(request: Request) -> AgentsClient:
    return request.app.state.agent_client


def get_agent(request: Request) -> Agent:
    return request.app.state.agent


def get_app_insights_conn_str(request: Request) -> str:
    if hasattr(request.app.state, "application_insights_connection_string"):
        return request.app.state.application_insights_connection_string
    else:
        return None


def serialize_sse_event(data: Dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def get_message_and_annotations(agent_client: AgentsClient, message: ThreadMessage) -> Dict:
    annotations = []
    # Get file annotations for the file search.
    for annotation in (a.as_dict() for a in message.file_citation_annotations):
        file_id = annotation["file_citation"]["file_id"]
        logger.info(f"Fetching file with ID for annotation {file_id}")
        openai_file = await agent_client.files.get(file_id)
        annotation["file_name"] = openai_file.filename
        logger.info(f"File name for annotation: {annotation['file_name']}")
        annotations.append(annotation)

    # Get url annotation for the index search.
    for url_annotation in message.url_citation_annotations:
        annotation = url_annotation.as_dict()
        annotation["file_name"] = annotation['url_citation']['title']
        logger.info(f"File name for annotation: {annotation['file_name']}")
        annotations.append(annotation)

    return {
        'content': message.text_messages[0].text.value,
        'annotations': annotations
    }


class MyEventHandler(AsyncAgentEventHandler[str]):
    def __init__(self, ai_project: AIProjectClient, app_insights_conn_str: str):
        super().__init__()
        self.agent_client = ai_project.agents
        self.ai_project = ai_project
        self.app_insights_conn_str = app_insights_conn_str

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        stream_data = {'content': delta.text, 'type': "message chunk"}
        # return serialize_sse_event(stream_data)
        return None

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        try:
            logger.info(f"MyEventHandler: Received thread message, message ID: {message.id}, status: {message.status}")
            if message.status != "completed":
                return None

            logger.info("MyEventHandler: Received completed message")

            stream_data = await get_message_and_annotations(self.agent_client, message)
            stream_data['type'] = "message"
            return serialize_sse_event(stream_data)
        except Exception as e:
            logger.error(f"Error in event handler for thread message: {e}", exc_info=True)
            return None

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        logger.info("MyEventHandler: on_thread_run event received")
        run_information = f"ThreadRun status: {run.status}, thread ID: {run.thread_id}"
        stream_data = {'content': run_information, 'type': 'thread_run'}
        if run.status == "failed":
            stream_data['error'] = run.last_error.as_dict()
        # automatically run agent evaluation when the run is completed
        if run.status == "completed":
            run_agent_evaluation(run.thread_id, run.id, self.ai_project, self.app_insights_conn_str)
        return serialize_sse_event(stream_data)

    async def on_error(self, data: str) -> Optional[str]:
        logger.error(f"MyEventHandler: on_error event received: {data}")
        stream_data = {'type': "stream_end"}
        return serialize_sse_event(stream_data)

    async def on_done(self) -> Optional[str]:
        logger.info("MyEventHandler: on_done event received")
        stream_data = {'type': "stream_end"}
        return serialize_sse_event(stream_data)

    async def on_run_step(self, step: RunStep) -> Optional[str]:
        logger.info(f"Step {step['id']} status: {step['status']}")
        step_details = step.get("step_details", {})
        tool_calls = step_details.get("tool_calls", [])

        if tool_calls:
            logger.info("Tool calls:")
            for call in tool_calls:
                azure_ai_search_details = call.get("azure_ai_search", {})
                if azure_ai_search_details:
                    logger.info(f"azure_ai_search input: {azure_ai_search_details.get('input')}")
                    logger.info(f"azure_ai_search output: {azure_ai_search_details.get('output')}")
        return None


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, _=auth_dependency):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        }
    )


async def get_result(
    request: Request,
    thread_id: str,
    agent_id: str,
    ai_project: AIProjectClient,
    app_insight_conn_str: Optional[str],
    carrier: Dict[str, str]
) -> AsyncGenerator[str, None]:
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    with tracer.start_as_current_span('get_result', context=ctx):
        logger.info(f"get_result invoked for thread_id={thread_id} and agent_id={agent_id}")
        try:
            agent_client = ai_project.agents
            async with await agent_client.runs.stream(
                thread_id=thread_id,
                agent_id=agent_id,
                event_handler=MyEventHandler(ai_project, app_insight_conn_str),
            ) as stream:
                logger.info("Successfully created stream; starting to process events")
                async for event in stream:
                    _, _, event_func_return_val = event
                    logger.debug(f"Received event: {event}")
                    if event_func_return_val:
                        logger.info(f"Yielding event: {event_func_return_val}")
                        yield event_func_return_val
                    else:
                        logger.debug("Event received but no data to yield")
        except Exception as e:
            logger.exception(f"Exception in get_result: {e}")
            yield serialize_sse_event({'type': "error", 'message': str(e)})


@router.get("/chat/history", summary="Get chat conversation history")
async def history(
    request: Request,
    ai_project: AIProjectClient = Depends(get_ai_project),
    agent: Agent = Depends(get_agent),
    _=auth_dependency
):
    with tracer.start_as_current_span("chat_history"):
        # Retrieve the thread ID from the cookies (if available).
        thread_id = request.cookies.get('thread_id')
        agent_id = request.cookies.get('agent_id')

        # Attempt to get an existing thread. If not found, create a new one.
        try:
            agent_client = ai_project.agents
            if thread_id and agent_id == agent.id:
                logger.info(f"Retrieving thread with ID {thread_id}")
                thread = await agent_client.threads.get(thread_id)
            else:
                logger.info("Creating a new thread")
                thread = await agent_client.threads.create()
        except Exception as e:
            logger.error(f"Error handling thread: {e}")
            raise HTTPException(status_code=400, detail=f"Error handling thread: {e}")

        thread_id = thread.id
        agent_id = agent.id

    # Create a new message from the user's input.
    try:
        content = []
        response = agent_client.messages.list(
            thread_id=thread_id,
        )
        async for message in response:
            formatteded_message = await get_message_and_annotations(agent_client, message)
            formatteded_message['role'] = message.role
            formatteded_message['created_at'] = message.created_at.astimezone().strftime("%m/%d/%y, %I:%M %p")
            content.append(formatteded_message)

        logger.info(f"List message, thread ID: {thread_id}")
        response = JSONResponse(content=content)

        # Update cookies to persist the thread and agent IDs.
        response.set_cookie("thread_id", thread_id)
        response.set_cookie("agent_id", agent_id)
        return response
    except Exception as e:
        logger.error(f"Error listing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error list message: {e}")


@router.get("/agent", summary="Get AI agent information")
async def get_chat_agent(
    request: Request
):
    return JSONResponse(content=get_agent(request).as_dict())


@router.post(
    "/chat",
    summary="Send message to AI agent",
    description="Send a message to the AI agent and receive a streaming response",
    responses={
        200: {
            "description": "Server-Sent Events stream with agent response",
            "content": {"text/event-stream": {"example": "data: {\"content\": \"Hello!\", \"type\": \"message\"}\n\n"}}
        },
        400: {"description": "Bad request - invalid input"},
        500: {"description": "Internal server error"}
    }
)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    ai_project: AIProjectClient = Depends(get_ai_project),
    app_insights_conn_str: str = Depends(get_app_insights_conn_str),
        _=auth_dependency
):
    """
    Send a message to the AI agent and receive a streaming response.

    - **message**: The user's message to send to the AI agent

    Returns a Server-Sent Events (SSE) stream with the agent's response.
    """
    # Retrieve the thread ID from the cookies (if available).
    thread_id = request.cookies.get('thread_id')
    agent_id = request.cookies.get('agent_id')

    with tracer.start_as_current_span("chat_request"):
        carrier = {}
        TraceContextTextMapPropagator().inject(carrier)

        # Attempt to get an existing thread. If not found, create a new one.
        try:
            agent_client = ai_project.agents
            if thread_id and agent_id == agent.id:
                logger.info(f"Retrieving thread with ID {thread_id}")
                thread = await agent_client.threads.get(thread_id)
            else:
                logger.info("Creating a new thread")
                thread = await agent_client.threads.create()
        except Exception as e:
            logger.error(f"Error handling thread: {e}")
            raise HTTPException(status_code=400, detail=f"Error handling thread: {e}")

        thread_id = thread.id
        agent_id = agent.id

        logger.info(f"user_message: {chat_request.message}")

        # Create a new message from the user's input.
        try:
            message = await agent_client.messages.create(
                thread_id=thread_id,
                role="user",
                content=chat_request.message
            )
            logger.info(f"Created message, message ID: {message.id}")
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating message: {e}")

        # Set the Server-Sent Events (SSE) response headers.
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
        logger.info(f"Starting streaming response for thread ID {thread_id}")

        # Create the streaming response using the generator.
        response = StreamingResponse(get_result(request, thread_id, agent_id, ai_project,
                                     app_insights_conn_str, carrier), headers=headers)

        # Update cookies to persist the thread and agent IDs.
        response.set_cookie("thread_id", thread_id)
        response.set_cookie("agent_id", agent_id)
        return response


@router.post("/api/send_payload", summary="sending monitoring payload to AI Agent")
async def process_payload(
    request: Request,
    payload: MonitorPayload = Body(..., description="The payload message request"),
    agent: Agent = Depends(get_agent),
    ai_project: AIProjectClient = Depends(get_ai_project),
    app_insights_conn_str: str = Depends(get_app_insights_conn_str)
):
    """
    Test endpoint that logs all incoming parameter values for debugging purposes.
    """
    try:
        # Log basic request information
        logger.debug(f"=== write_parameters_to_log called ===")
        logger.debug(f"Request method: {request.method}")
        logger.debug(f"Request URL: {request.url}")
        logger.debug(f"Request headers: {dict(request.headers)}")

        # Log cookies
        if request.cookies:
            logger.debug(f"Request cookies: {dict(request.cookies)}")
        else:
            logger.debug("No cookies in request")

        request_data = await request.json()
        logger.debug(f"Request body: {request_data}")

        # Log the chat request body
        logger.debug(f"Payload: {payload.model_dump()}")

        # Log agent information
        if agent:
            logger.debug(f"Agent ID: {agent.id}")
            logger.debug(f"Agent name: {agent.name}")
            logger.debug(f"Agent model: {agent.model}")
        else:
            logger.info("No agent available")

        # Log AI project information
        if ai_project:
            logger.info(f"AI Project type: {type(ai_project)}")
            logger.info(f"AI Project endpoint: {getattr(ai_project, '_endpoint', 'Unknown')}")
        else:
            logger.info("No AI project available")

        # Log app insights connection string (safely)
        if app_insights_conn_str:
            # Only log first and last 10 characters for security
            safe_conn_str = f"{app_insights_conn_str[:10]}...{app_insights_conn_str[-10:]}" if len(
                app_insights_conn_str) > 20 else "[REDACTED]"
            logger.info(f"App Insights connection string (partial): {safe_conn_str}")
        else:
            logger.info("No App Insights connection string")

        # Log client information
        client_host = request.client.host if request.client else "Unknown"
        client_port = request.client.port if request.client else "Unknown"
        logger.debug(f"Client: {client_host}:{client_port}")

        # Log query parameters if any
        if request.query_params:
            logger.debug(f"Query parameters: {dict(request.query_params)}")
        else:
            logger.debug("No query parameters")

        logger.debug(f"=== End of parameter logging ===")

        if (hasattr(payload, 'body') and hasattr(payload.body, 'data')):
            payload_data = payload.body.data
            if hasattr(payload_data, 'essentials') and hasattr(payload_data.essentials, 'alertTargetIDs'):
                targetIDs = payload_data.essentials.alertTargetIDs
            else:
                logger.warning("AlertTargetIDs not found in payload: %s", payload)
                targetIDs = []
        else:
            raise ValueError("Invalid payload structure")

        tags = []
        for targetID in targetIDs:
            id_tags = get_tags(targetID)
            tags.append({f"{targetID}": id_tags})

        metrics = []
        for targetID in targetIDs:
            if (hasattr(payload_data, 'alertContext') and hasattr(payload_data.alertContext, 'condition')
                    and hasattr(payload_data.alertContext.condition, 'allOf')):
                metric_names = []
                for m in payload_data.alertContext.condition.allOf:
                    metric_info = {
                        "metricName": m.metricName,
                        "metricNamespace": m.metricNamespace,
                        "metricValue": m.metricValue,
                        "threshold": m.threshold,
                        "operator": m.operator,
                        "timeAggregation": m.timeAggregation
                    }
                    metric_names.append(m.metricName)
                id_metrics = get_monitored_metrics(targetID, metric_info.get("metricNamespace", ""),
                                                   metric_names)
                metrics.append({f"{targetID}": id_metrics})

        # payload_dict = {**payload.model_dump(), "tags": tags, "metrics": metrics}
        # chat_request = ChatRequest(message=f'The payload representing the alert is {payload_dict}')
        chat_request = ChatRequest(message=f'The alert payload is {payload.model_dump()},'
                                   f' the resource tags are {tags}. The monitored metric readings for the past'
                                   f' 4 hour are \n{metrics}\n. '
                                   f'Please analyze the alert payload, incorporate the meaning of the tags '
                                   f'and the recent metric reading, and provide possible reasons, '
                                   'if the issue persists, or is an anomaly and remediation suggestions.')
        agent_response = await chat(request, chat_request, get_agent(request), get_ai_project(request))

        agent_response_body = [str(section) async for section in agent_response.body_iterator]

        return_message = ""
        for data in agent_response_body:
            data_obj = json.loads(data.removeprefix("data: "))
            if data_obj.get('type') == 'message':
                return_message += (data_obj.get('content', '') + "\n")
                logger.info(f"Agent response message: {data_obj.get('content')}")
        import markdown
        return_message_html = markdown.markdown(return_message)
        return (return_message_html.replace("\n", "<br>"))
        # return JSONResponse({
        #     "status": "success",
        #     "message": "All parameters logged successfully",
        #     "timestamp": str(asyncio.get_event_loop().time()),
        #     "logged_items": [
        #         "request_info",
        #         "cookies",
        #         "chat_request",
        #         "agent_info",
        #         "ai_project_info",
        #         "app_insights_connection",
        #         "client_info",
        #         "query_params"
        #     ]
        # })

    except Exception as e:
        logger.error(f"Error in write_parameters_to_log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error logging parameters: {str(e)}")


def read_file(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()


def run_agent_evaluation(
        thread_id: str,
        run_id: str,
        ai_project: AIProjectClient,
        app_insights_conn_str: str):

    if app_insights_conn_str:
        agent_evaluation_request = AgentEvaluationRequest(
            run_id=run_id,
            thread_id=thread_id,
            evaluators={
                "Relevance": {"Id": EvaluatorIds.RELEVANCE.value},
                "TaskAdherence": {"Id": EvaluatorIds.TASK_ADHERENCE.value},
                "ToolCallAccuracy": {"Id": EvaluatorIds.TOOL_CALL_ACCURACY.value},
            },
            sampling_configuration=AgentEvaluationSamplingConfiguration(
                name="default",
                sampling_percent=100,
            ),
            redaction_configuration=AgentEvaluationRedactionConfiguration(
                redact_score_properties=False,
            ),
            app_insights_connection_string=app_insights_conn_str,
        )

        async def run_evaluation():
            try:
                logger.info(f"Running agent evaluation on thread ID {thread_id} and run ID {run_id}")
                agent_evaluation_response = await ai_project.evaluations.create_agent_evaluation(
                    evaluation=agent_evaluation_request
                )
                logger.info(f"Evaluation response: {agent_evaluation_response}")
            except Exception as e:
                logger.error(f"Error creating agent evaluation: {e}")

        # Create a new task to run the evaluation asynchronously
        asyncio.create_task(run_evaluation())


@router.get("/config/azure", summary="Get Azure configuration")
async def get_azure_config(_=auth_dependency):
    """Get Azure configuration for frontend use"""
    try:
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        tenant_id = os.environ.get("AZURE_TENANT_ID", "")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "")
        ai_project_resource_id = os.environ.get("AZURE_EXISTING_AIPROJECT_RESOURCE_ID", "")

        # Extract resource name and project name from the resource ID
        # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{resource}/projects/{project}
        resource_name = ""
        project_name = ""

        if ai_project_resource_id:
            parts = ai_project_resource_id.split("/")
            if len(parts) >= 8:
                resource_name = parts[8]  # accounts/{resource_name}
            if len(parts) >= 10:
                project_name = parts[10]  # projects/{project_name}

        return JSONResponse({
            "subscriptionId": subscription_id,
            "tenantId": tenant_id,
            "resourceGroup": resource_group,
            "resourceName": resource_name,
            "projectName": project_name,
            "wsid": ai_project_resource_id
        })
    except Exception as e:
        logger.error(f"Error getting Azure config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Azure configuration")
