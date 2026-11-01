"""The AzSelector Lambda (inline Python in cdk/lib/az-selector.ts): On-Demand first, then one-time Spot."""
import re
import sys
import types
from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "cdk" / "lib" / "az-selector.ts").read_text()
CODE = re.search(r"const AZ_SELECTOR_LAMBDA_CODE = `\n(.*?)\n`;", SRC, re.S).group(1)


class EC2:
    def __init__(self, free):
        self.free = free  # set of (market, type)
        self.calls = []

    def describe_subnets(self, SubnetIds):
        return {"Subnets": [{"SubnetId": s, "AvailabilityZone": f"az-{s}"} for s in SubnetIds]}

    def describe_instance_type_offerings(self, **_):
        return {"InstanceTypeOfferings": [{"Location": "az-s1"}, {"Location": "az-s2"}]}

    def run_instances(self, **kw):
        market = kw.get("InstanceMarketOptions", {}).get("MarketType", "on-demand")
        self.calls.append((market, kw["InstanceType"]))
        if (market, kw["InstanceType"]) not in self.free:
            raise Exception("An error occurred (InsufficientInstanceCapacity) when calling RunInstances")
        return {"Instances": [{"InstanceId": "i-1"}]}


def run(free, spot):
    sent = []
    cfn = types.SimpleNamespace(SUCCESS="SUCCESS", FAILED="FAILED",
                                send=lambda e, c, status, data, **kw: sent.append((status, data, kw)))
    ec2 = EC2(free)
    sys.modules["cfnresponse"] = cfn
    sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: ec2)
    try:
        ns = {}
        exec(compile(CODE, "az_selector", "exec"), ns)  # nosec B102 - the repo's own Lambda source
        event = {"RequestType": "Create", "ResourceProperties": {
            "InstanceTypes": "p5.48xlarge,p4de.24xlarge", "AmiId": "ami-1", "SubnetIds": "s1,s2",
            "UserDataBase64": "", "IamInstanceProfileArn": "arn", "SecurityGroupIds": "sg-1", "EbsGb": "100",
            "Tags": "[]", "SpotFallback": "true" if spot else "false"}}
        ns["handler"](event, types.SimpleNamespace(invoked_function_arn="arn:aws:lambda:us-west-2:1:function:f"))
    finally:
        del sys.modules["cfnresponse"], sys.modules["boto3"]
    return sent[-1], ec2.calls


def test_on_demand_wins_before_any_spot_try():
    (status, data, _), calls = run({("on-demand", "p4de.24xlarge")}, spot=True)
    assert status == "SUCCESS" and data["Market"] == "on-demand" and data["InstanceType"] == "p4de.24xlarge"
    assert all(m == "on-demand" for m, _ in calls)


def test_spot_after_every_on_demand_type_failed():
    (status, data, _), calls = run({("spot", "p5.48xlarge")}, spot=True)
    assert status == "SUCCESS" and data["Market"] == "spot" and data["InstanceType"] == "p5.48xlarge"
    assert [m for m, _ in calls[:4]] == ["on-demand"] * 4 and calls[-1] == ("spot", "p5.48xlarge")


def test_no_spot_without_the_flag():
    (status, _, kw), calls = run({("spot", "p5.48xlarge")}, spot=False)
    assert status == "FAILED" and "No capacity" in kw["reason"]
    assert all(m == "on-demand" for m, _ in calls)
