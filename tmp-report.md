# Benchmark Report: vulhub-struts2-s2045-local

- Target ID: `vulhub-struts2-s2045-local`
- Target Name: Local Vulhub Struts2 S2-045
- Source: VulnHub
- Risk Level: low
- Ablation: none

## PentestGPT / AutoPentester Alignment Metrics

| Metric | Alignment | Value |
| --- | --- | ---: |
| Target Completion | PentestGPT task success / AutoPentester successful attack | true |
| Subtask Completion % | PentestGPT reasoning task completion | 100 |
| Service Coverage % | Reconnaissance/service discovery coverage | 100 |
| Vulnerability Coverage % | Validated vulnerability discovery coverage | 100 |
| Steps | PentestGPT action count / AutoPentester rounds | 2 |
| Loops | Repeated task/command/phase count | 0 |
| Human Interaction | Manual approval or intervention count | 0 |
| Incomplete Commands | Failed, placeholder, or unsupported command count | 0 |
| Time | Elapsed/runtime seconds | 3.5 |
| Cost | LLM/tool cost when exported | null |
| KG Node Recall | Aegra KG node recall against ground truth | 100 |
| KG Edge Recall | Aegra KG edge recall against ground truth | 100 |
| Evidence Chain Completeness | Host -> Service -> Vulnerability -> Evidence chain coverage | 100 |
| False Positive Rate | Unmatched or rejected finding ratio | 0 |

## Matched Ground Truth

| Category | IDs |
| --- | --- |
| Services | svc-http-8080 |
| Vulnerabilities | vuln-struts2-s2045 |
| Subtasks | subtask-reachability, subtask-vuln-validation |
