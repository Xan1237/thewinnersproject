#!/usr/bin/env python3
"""Small standard-library HTTP adapter suitable for an LLM tool handler."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request


class RobotApi:
    def __init__(self, base_url="http://127.0.0.1:8817", token=None):
        self.base_url = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("ROBOT_API_TOKEN", "")

    def request(self, path, data=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.base_url + path,
            data=None if data is None else json.dumps(data).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Robot API {error.code}: {error.read().decode()}") from None

    def state(self):
        return self.request("/api/state")

    def schema(self):
        return self.request("/api/schema")

    def command(self, command):
        return self.request("/api/commands", command)

    def stop(self):
        return self.request("/api/stop", {})

    def wait(self, command_id, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.request(f"/api/commands/{command_id}")
            if result.get("status") in {"completed", "rejected"}:
                return result
            time.sleep(0.1)
        raise TimeoutError("Command has not finished; inspect state or send stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["state", "schema", "command", "stop"])
    parser.add_argument("payload", nargs="?", help="Command JSON")
    parser.add_argument("--url", default="http://127.0.0.1:8817")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    api = RobotApi(args.url)
    if args.operation == "command":
        if not args.payload:
            parser.error("command requires a JSON payload")
        result = api.command(json.loads(args.payload))
        if args.wait:
            result = api.wait(result["id"])
    else:
        result = getattr(api, args.operation)()
    print(json.dumps(result, indent=2, ensure_ascii=False))
