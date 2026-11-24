import os, json, boto3
from dotenv import load_dotenv
load_dotenv()

region = os.getenv("AWS_REGION")
key = os.getenv("AWS_ACCESS_KEY_ID") or ""
print("Region     :", region)
print("Key prefix :", key[:4], "(AKIA = long-term IAM key, ASIA = temporary)")

sts = boto3.client(
    "sts",
    region_name=region,
    aws_access_key_id=key,
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
)
ident = sts.get_caller_identity()
print("Account    :", ident["Account"])
print("Identity   :", ident["Arn"])

rt = boto3.client(
    "bedrock-runtime",
    region_name=region,
    aws_access_key_id=key,
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
)
try:
    r = rt.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": "hello", "dimensions": 1024, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    print("Bedrock OK, dims:", len(json.loads(r["body"].read())["embedding"]))
except Exception as e:
    print("Bedrock FAILED:", e)