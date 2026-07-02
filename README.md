# Dynamo

Dynamo is a compound AI system architecture for coding tasks.

## Architecture

Dynamo uses a two-stage design:

1. **Router** — A general-purpose model that interprets user intent and directs requests.
2. **CodeCore** — A specialized model trained exclusively on code. It receives routed requests and handles all code generation, editing, and reasoning tasks.

The router handles language understanding; CodeCore handles execution. Neither does the other's job.
