from datetime import timedelta
from azure.identity import DefaultAzureCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest, QueryResponse

from azure.monitor.querymetrics import MetricsClient, MetricAggregationType
from azure.mgmt.monitor import MonitorManagementClient


_cred = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
_rgc = ResourceGraphClient(credential=_cred)


def query_resource(TargetID: str, project: str = "tags, location") -> QueryResponse:
    """Returns the tags for the VM via Azure Resource Graph."""
    # Extract subscription ID from TargetID
    # TargetID format: /subscriptions/{subscription-id}/resourcegroups/{rg}/providers/{provider}/{resource}
    try:
        parts = TargetID.split('/')
        if len(parts) >= 3 and parts[1] == 'subscriptions':
            sub = parts[2]
        else:
            raise ValueError("Invalid TargetID format - missing subscription ID")
    except (IndexError, ValueError) as e:
        print(f"Error extracting subscription ID from TargetID: {e}")
        return QueryResponse(total_records=0, count=0, result_truncated="False", data=[])

    query = (f'resources | where id =~ "{TargetID}" | project tags, location')

    req = QueryRequest(subscriptions=[sub], query=query)
    resp = _rgc.resources(req)
    return resp


def get_tags(TargetID: str) -> dict:
    resp_data = query_resource(TargetID).data
    try:
        if isinstance(resp_data, list):
            return resp_data[0].get("tags", {}) if resp_data else {}
        elif isinstance(resp_data, dict):
            return resp_data.get("tags", {})
        else:
            return {}
    except Exception:
        return {}


def get_monitored_metrics(TargetID: str, metric_namespace: str = "microsoft",
                          metric_names: list[str] = []) -> str:
    """Returns the monitored metrics for the VM via Azure Resource Graph."""
    # Extract subscription ID from TargetID
    # TargetID format: /subscriptions/{subscription-id}/resourcegroups/{rg}/providers/{provider}/{resource}
    try:
        parts = TargetID.split('/')
        if len(parts) >= 3 and parts[1] == 'subscriptions':
            sub = parts[2]
        else:
            raise ValueError("Invalid TargetID format - missing subscription ID")
    except (IndexError, ValueError) as e:
        print(f"Error extracting subscription ID from TargetID: {e}")
        return ""

    # _mmc = MonitorManagementClient(credential=_cred, subscription_id=sub)
    # definitions = _mmc.metric_definitions.list(TargetID)

    response_data = query_resource(TargetID, project="location").data
    loc = ""
    if isinstance(response_data, list) and response_data:
        loc = response_data[0].get("location", "")
    endpoint = f"https://{loc}.metrics.monitor.azure.com"
    _mc = MetricsClient(credential=_cred, endpoint=endpoint)

    resp_data = _mc.query_resources(
        resource_ids=[TargetID],
        metric_namespace=metric_namespace,
        metric_names=metric_names,
        timespan=timedelta(hours=4),
        granularity=timedelta(minutes=5),
        aggregations=[MetricAggregationType.AVERAGE, MetricAggregationType.MAXIMUM, MetricAggregationType.MINIMUM,
                      MetricAggregationType.COUNT, MetricAggregationType.TOTAL]
    )
    try:
        metric = resp_data[0].metrics[0]
        timeseries_data = metric.timeseries[0].data
        ts_schema = []
        ts_schema.append(f"base_ts={timeseries_data[0].timestamp}; dt_ms=ms_since_base")
        for attr in ["average", "maximum", "minimum", "count", "total"]:
            if metric.unit == "Percent":
                ts_schema.append(f"{attr}=scale=100;pct")
            elif metric.unit == "Count":
                ts_schema.append(f"{attr}=scale=1;count")
            else:
                ts_schema.append(f"{attr}=cat")

        schema_str = "SCHEMA: " + "; ".join(ts_schema)
        stat_str = (f'STATS: n={len(timeseries_data)}; range={resp_data[0].timespan};'
                    f' average_avg={sum(getattr(ts, "average", 0) for ts in timeseries_data) / len(timeseries_data) if timeseries_data else 0}'
                    f' count_total={sum(getattr(ts, "count", 0) for ts in timeseries_data)}'
                    f' maximum_max={max(getattr(ts, "maximum", 0) for ts in timeseries_data)}'
                    f' minimum_min={min(getattr(ts, "minimum", 0) for ts in timeseries_data)}'
                    f' total_sum={sum(getattr(ts, "total", 0) for ts in timeseries_data)}')
        data_array = []
        for ts in timeseries_data:
            entry = []
            value = int((ts.timestamp - timeseries_data[0].timestamp).total_seconds() * 1000)
            entry.append(str(value))
            for attr in ["average", "maximum", "minimum", "count", "total"]:
                value = getattr(ts, attr, "")
                if metric.unit == "Percent" and isinstance(value, (int, float)):
                    value = int(value * 100)
                entry.append(str(value))
            data_array.append(entry)
        data_str = "DATA: \n" + "\n".join([", ".join(entry) for entry in data_array])
        return f"{schema_str}\n{stat_str}\n{data_str}"
    except Exception:
        return ""
