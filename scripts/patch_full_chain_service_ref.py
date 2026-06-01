from pathlib import Path

p = Path("/tmp/test_full_real_chain_vulhub_12615.py")
lines = p.read_text(encoding="utf-8").splitlines()

inserted = False

for i, line in enumerate(lines):
    if "vuln_ref = KGGraphRef" in line and "VULN_ID" in line:
        window = lines[max(0, i - 5):i]
        if any("service_ref = KGGraphRef" in item for item in window):
            print("service_ref already exists near vuln_ref")
            inserted = True
            break

        indent = line[: len(line) - len(line.lstrip())]
        service_line = (
            indent
            + 'service_ref = KGGraphRef(graph="kg", ref_id=SERVICE_ID, '
            + 'ref_type="Service", label="Vulhub Tomcat 8081")'
        )
        lines.insert(i, service_line)
        print(f"inserted service_ref before line {i + 1}")
        inserted = True
        break

if not inserted:
    print("target vuln_ref line not found; showing related lines:")
    for idx, line in enumerate(lines, start=1):
        if "vuln_ref" in line or "service_ref" in line:
            print(f"{idx}: {line}")
    raise SystemExit(1)

p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("patched /tmp/test_full_real_chain_vulhub_12615.py")