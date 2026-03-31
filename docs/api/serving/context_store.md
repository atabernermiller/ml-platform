# Context Store

Context stores correlate predictions with delayed feedback. When `predict()` fires, the service stores feature context keyed by `request_id`. When `process_feedback()` arrives later, the context is retrieved so the model can learn from the (context, reward) pair.

## ContextStore (ABC)

::: ml_platform.serving.context_store.ContextStore

## DynamoDBContextStore

::: ml_platform.serving.context_store.DynamoDBContextStore

## InMemoryContextStore

::: ml_platform.serving.context_store.InMemoryContextStore
