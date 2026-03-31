# CDK Constructs

AWS CDK constructs for deploying ml-platform services. All constructs are designed to compose together — `NetworkConstruct` provisions the VPC, then `EcsServiceConstruct` and `MLflowConstruct` deploy into it, and `MonitoringConstruct` sets up alarms tied to the same `MLPlatform` CloudWatch namespace used by `MetricsEmitter`.

## NetworkConstruct

::: ml_platform.infra.constructs.network.NetworkConstruct

## EcsServiceConstruct

::: ml_platform.infra.constructs.ecs_service.EcsServiceConstruct

## MLflowConstruct

::: ml_platform.infra.constructs.mlflow.MLflowConstruct

## MonitoringConstruct

::: ml_platform.infra.constructs.monitoring.MonitoringConstruct

## SageMakerEndpointConstruct

::: ml_platform.infra.constructs.sagemaker.SageMakerEndpointConstruct
