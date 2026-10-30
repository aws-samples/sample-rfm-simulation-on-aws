"""deploy._maybe_import_orphan_bucket: a finished run's stack is replaced, a live run is not touched."""
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deploy  # noqa: E402


class _Waiter:
    def wait(self, **_):
        pass


def _clients(stack_status, instance_state, calls):
    cfn = mock.Mock()
    cfn.describe_stacks.return_value = {"Stacks": [{
        "StackStatus": stack_status,
        "Outputs": [{"OutputKey": "InstanceId", "OutputValue": "i-0abc"}],
    }]}
    cfn.delete_stack.side_effect = lambda **kw: calls.append(("delete_stack", kw["StackName"]))
    cfn.get_waiter.return_value = _Waiter()
    ec2 = mock.Mock()
    ec2.describe_instances.return_value = {"Reservations": [{"Instances": [{"State": {"Name": instance_state}}]}]}
    sts = mock.Mock()
    sts.get_caller_identity.return_value = {"Account": "123456789012"}
    s3 = mock.Mock()
    return {"cloudformation": cfn, "ec2": ec2, "sts": sts, "s3": s3}


def _run(stack_status, instance_state):
    calls = []
    clients = _clients(stack_status, instance_state, calls)
    with mock.patch.object(deploy.boto3, "client", side_effect=lambda name, **_: clients[name]), \
            mock.patch.object(deploy.subprocess, "run", side_effect=lambda cmd, **_: calls.append(("run", cmd[:3]))):
        deploy._maybe_import_orphan_bucket("cdk", "SmolVLA-Demo", "us-east-1", "a@b.c", "", "vla-sim-results", [])
    return calls


def test_finished_run_stack_is_deleted_then_bucket_readopted():
    calls = _run("CREATE_COMPLETE", "terminated")
    assert calls[0] == ("delete_stack", "SmolVLA-Demo")
    assert calls[1] == ("run", ["npx", "cdk", "import"])


def test_live_run_is_refused():
    with pytest.raises(SystemExit):
        _run("UPDATE_COMPLETE", "running")


def test_in_progress_stack_left_alone():
    assert _run("CREATE_IN_PROGRESS", "running") == []
