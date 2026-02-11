
"""
AWS DynamoDB tool for querying database resources
"""

import json
import boto3
from boto3.dynamodb.conditions import Key
from langchain_core.tools import tool
from core.models import AWSInput, DecimalEncoder
from core.config import AWS_CONFIG

# Initialize AWS session
awsSession = boto3.Session(
    aws_access_key_id=AWS_CONFIG["aws_access_key_id"],
    aws_secret_access_key=AWS_CONFIG["aws_secret_access_key"],
    aws_session_token=AWS_CONFIG["aws_session_token"],
    region_name=AWS_CONFIG["region_name"]
)
dynamodb = awsSession.resource("dynamodb")

@tool(args_schema=AWSInput)
def search_aws_cli(table_name, partition_key: str) -> str:
    """
    Query AWS DynamoDB resources for infrastructure and database data.
    Use this tool when you need information about campaigns, subscriptions, user data stored in DynamoDB.

    Table: prod-district-membership-service

    Partition key formats:
    - For campaigns with gold plan: PLAN_TYPE:PLAN_TYPE_DISTRICT_GOLD
    - For campaigns with pass plan: PLAN_TYPE:PLAN_TYPE_DISTRICT_GOLD_INDIA
    - For subscription/user data: USER:{user_id}

    Args:
        partition_key: DynamoDB partition key to query
        table_name: DynamoDB table name

    Returns:
        JSON string with queried DynamoDB data
    """
    table = dynamodb.Table(table_name)
    response = table.query(
        KeyConditionExpression=Key("partition_key").eq(partition_key)
    )
    items = response.get('Items', [])

    result = {
        "table": table_name,
        "partition_key": partition_key,
        "items_found": len(items),
        "data": json.loads(json.dumps(items, cls=DecimalEncoder))
    }

    return f"[AWS DynamoDB]\n{json.dumps(result, indent=2)}"
