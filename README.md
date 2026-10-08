# spec-review

Requirements as code: an LLM-assisted reviewer for engineering requirements. It checks
requirement quality against INCOSE and ISO/IEC/IEEE 29148 rules, classifies requirements, finds
conflicting and duplicate requirements, and recovers traceability links into a graph for impact
analysis. It runs as a GitHub Action on pull requests that change requirement files.

Runs on a CPU: open-weight models through llama.cpp, no hosted APIs.

Work in progress.
