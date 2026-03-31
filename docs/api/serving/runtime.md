# StatefulRuntime

`StatefulRuntime` owns the complete lifecycle of a `StatefulServiceBase` — state restoration, background checkpoint/metric loops, and graceful shutdown — without depending on any HTTP framework.

Use it directly for non-HTTP transports (gRPC, AWS Lambda, CLI tools) or let `create_stateful_app()` wire it into FastAPI automatically.

```python
from ml_platform.serving.runtime import StatefulRuntime
from ml_platform.config import ServiceConfig

runtime = StatefulRuntime(MyService, config)

await runtime.startup()
result = await runtime.predict({"context": [1, 2, 3]})
await runtime.process_feedback(result.request_id, {"reward": 1.0})
snapshot = runtime.metrics_snapshot()
await runtime.shutdown()
```

::: ml_platform.serving.runtime.StatefulRuntime
