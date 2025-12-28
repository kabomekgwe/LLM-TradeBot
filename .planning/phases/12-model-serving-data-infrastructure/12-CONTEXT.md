# Phase 12: Model Serving & Data Infrastructure - Context

**Gathered:** 2025-12-28
**Status:** Ready for research

<vision>
## How This Should Work

A complete production infrastructure for the final missing pieces: reliable model serving, persistent trade history, secure secrets management, and structured logging. This phase completes the production readiness journey.

The approach is **pragmatic and cohesive** - build on Phase 11's Docker infrastructure with battle-tested, open-source technologies. Not over-engineered cloud-native solutions, but not amateur hour either. Production-grade where it matters.

When this phase is done, the trading bot has everything needed for real production deployment: models serve predictions reliably via FastAPI, trade data persists in PostgreSQL + TimescaleDB, API keys are managed securely via Docker secrets, and structured JSON logs enable debugging without SSH'ing into containers.

</vision>

<essential>
## What Must Be Nailed

**All four components are equally critical** - this is a system, and failure in any component breaks production trading:

- **Model serving** - ML predictions drive trading decisions. Models must be fast, reliable, and never fail. Bad predictions or downtime directly cost money.
- **Trade history persistence** - Trade data must never be lost. Need backups, proper schema, time-series optimization. Without trade data, can't analyze performance or meet regulatory requirements.
- **Secrets management** - Exchange API keys control real money. Proper secrets handling is paramount. One leaked key = catastrophic loss.
- **Structured logging** - Production debugging requires proper logs. Can't troubleshoot issues without being able to query logs effectively.

No shortcuts, no "we'll improve this later" - all four must be production-ready on first deployment.

</essential>

<boundaries>
## What's Out of Scope

Phase 12 builds production infrastructure, not advanced features:

- **Multi-region deployment** - Single region deployment is sufficient. No geographic replication, edge deployment, or multi-region failover.
- Advanced features are acceptable if they serve production needs, but not required:
  - Model versioning, A/B testing, canary deployments (nice-to-have, but not required for Phase 12)
  - Real-time streaming, Kafka, event pipelines (can use batch processing and polling)
  - Advanced analytics dashboards, data warehousing (just store and query trade history)

Focus is on **production readiness**, not bells and whistles.

</boundaries>

<specifics>
## Specific Ideas

**Technology stack preferences** (pragmatic, builds on Phase 11):

- **PostgreSQL + TimescaleDB extension** - Battle-tested relational DB with time-series optimization for trade data. Open source, SQL familiar, scales well.
- **FastAPI model serving** - Serve models via existing FastAPI framework (same as dashboard from Phase 10). Simple, consistent, no new dependencies or frameworks to learn.
- **Docker secrets** - Use Docker's native secrets management for API keys. Already have Docker infrastructure from Phase 11, integrates naturally.
- **Structured JSON logging** - Use python-json-logger (already in requirements.txt from early phases). Works with Docker logs, easy future integration with CloudWatch/Grafana if needed.

**Key design principle**: Build on what we have (Docker, FastAPI, PostgreSQL) rather than introducing new complexity. Production-grade doesn't mean enterprise complexity - it means reliable, maintainable, and well-integrated.

</specifics>

<notes>
## Additional Context

Phase 12 is the final piece of v1.2 milestone "Production Deployment & Live Trading". After this phase:
- Phase 9: Emergency safety controls ✅
- Phase 10: Real-time monitoring infrastructure ✅
- Phase 11: Dockerized production deployment ✅
- Phase 12: Model serving & data infrastructure (current)

This phase completes the production readiness journey - everything needed to deploy the trading bot to production with confidence.

**Approach clarification**: Initially suggested "hybrid approach" (some production-grade, some simple), but upon discussion, all four components are critical and need production-grade treatment. The "pragmatic" part is choosing simple, cohesive technologies (PostgreSQL vs managed cloud DB, FastAPI vs TorchServe) rather than the quality bar.

</notes>

---

*Phase: 12-model-serving-data-infrastructure*
*Context gathered: 2025-12-28*
