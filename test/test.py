import boto3

region = "us-east-1"  # use the same region as your console
print("Identity:", boto3.client("sts").get_caller_identity()["Arn"])
print("Account :", boto3.client("sts").get_caller_identity()["Account"])

bedrock = boto3.client("bedrock", region_name=region)
models = bedrock.list_foundation_models(byOutputModality="EMBEDDING")
print([m["modelId"] for m in models["modelSummaries"]])

rt = boto3.client("bedrock-runtime", region_name=region)
import json
try:
    r = rt.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": "hello", "dimensions": 1024, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    print("OK, dims:", len(json.loads(r["body"].read())["embedding"]))
except Exception as e:
    print("FAILED:", e)