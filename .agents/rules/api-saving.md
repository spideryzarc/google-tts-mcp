---
trigger: always_on
glob: ""
description: Guidelines to avoid unauthorized or redundant live API consumption and quota usage during testing and execution.
---

# API Cost & Quota Protection Guidelines

**Objective**: Protect API quota, prevent unnecessary cloud usage costs, and avoid rate-limiting or accidental usage of live external endpoints (e.g., Google Cloud Text-to-Speech API).

## Core Principles

1. **No Unsanctioned Live API Calls**
   - NEVER make live external API requests during routine development, code verification, linting, or unit testing unless explicitly requested by the user.

2. **Mocking & Offline Verification First**
   - Use unit tests with mocks, stubs, fixtures, or synthetic response objects for code validation.
   - Verify logic flow, error handling, parameter parsing, and state handling using local mocks without reaching out to external networks.

3. **Pre-Execution Safety Verification**
   - Before running test suites or execution scripts, inspect them to confirm they run strictly in offline/mock mode.
   - If a script or test file relies on live API calls without existing mocks, modify or isolate the test to use mocks before running it.

4. **Explicit Authorization Protocol**
   - Live API calls are permitted ONLY when the user explicitly instructs to run live integration tests against real APIs.
   - When executing explicitly requested live calls, keep request payloads as minimal as possible to conserve quota.

