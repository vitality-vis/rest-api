"""
Backend service package organized by architectural layer.

Subpackages:
- `service.api`: delivery-facing interfaces
- `service.core`: cross-cutting core types and policies
- `service.application`: use-case orchestration and query pipeline
- `service.agents`: agent runtime and orchestration actors
- `service.tools`: executable tool layer
- `service.integrations`: external provider adapters
- `service.infrastructure`: infrastructure adapters such as embeddings and vector stores
- `service.memory`: in-process session and memory state
- `service.writing`: grounded writing helpers
- `service.utils`: internal utility helpers
"""
