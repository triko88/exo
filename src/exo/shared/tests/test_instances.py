import json
from pydantic import TypeAdapter
from exo.shared.types.common import Host
from exo.shared.types.worker.instances import Instance, TinygradCPUInstance, InstanceId
from exo.shared.types.worker.runners import ShardAssignments

def test_tinygrad_instance_serialization():
    instance = TinygradCPUInstance(
        instance_id=InstanceId("test-tinygrad"),
        shard_assignments=ShardAssignments(
            model_id="test-model",
            runner_to_shard={},
            node_to_runner={}
        ),
        hosts=[Host(ip="127.0.0.1", port=8000)]
    )

    # Serialize
    json_str = instance.model_dump_json()
    data = json.loads(json_str)
    
    assert "TinygradCPUInstance" in data
    inner = data["TinygradCPUInstance"]
    assert inner["instanceId"] == "test-tinygrad"
    
    # Roundtrip through Instance Union
    adapter = TypeAdapter(Instance)
    restored = adapter.validate_json(json_str)
    
    assert isinstance(restored, TinygradCPUInstance)
    assert restored.instance_id == "test-tinygrad"
    assert len(restored.hosts) == 1
    assert restored.hosts[0].ip == "127.0.0.1"
    assert restored.hosts[0].port == 8000
