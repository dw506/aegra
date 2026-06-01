from pathlib import Path

p = Path("/tmp/mcp_vulhub_lab_server.py")
s = p.read_text(encoding="utf-8")

needle = "    return result\n\n\nclass Handler"

replacement = (
    "    result.setdefault(\"success\", True)\n"
    "    result.setdefault(\"stderr\", \"\")\n"
    "    result.setdefault(\"exit_code\", 0)\n"
    "    result.setdefault(\"stdout\", json.dumps(result, ensure_ascii=False))\n"
    "    return result\n\n\nclass Handler"
)

if needle not in s:
    print("needle not found, show return result lines:")
    for i, line in enumerate(s.splitlines(), start=1):
        if "return result" in line:
            print(f"{i}: {line}")
    raise SystemExit(1)

s = s.replace(needle, replacement, 1)
p.write_text(s, encoding="utf-8")
print("patched MCP server return fields")