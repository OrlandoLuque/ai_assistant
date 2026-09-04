#!/usr/bin/env python3
"""
V302 — OpenAPI-vs-routes gate.

Fails when the embedded server serves a route that `/openapi.json` does not declare, or
declares one it does not serve.

Why the existing unit test could not do this. `test_server_api_spec_has_all_endpoints`
asserts that a hand-written list of paths is present in the spec. That checks the paths
someone REMEMBERED to add, and by construction cannot detect the opposite — a route that
exists and is undeclared. It passed happily while ten endpoints were missing (V301).

The asymmetry matters more than it looks. A spec that promises endpoints that do not
exist fails loudly the first time a client calls one. A spec that hides endpoints that do
exist never fails at all: the generated client simply lacks them, and nobody involved has
any way to notice. That is the failure mode this catches.

Usage:
    python scripts/check_openapi_routes.py

Exits 1 if the two sides disagree.

WHAT IT PARSES, and the limits of that. Routes come from the `("METHOD", "/path")` match
arms in `src/server.rs`; declared paths from the `"paths": { ... }` map in
`generate_server_api_spec`. Both are read as text, so a route registered some other way is
invisible here — `/ws` is exactly that case (a WebSocket upgrade checked before the match)
and is listed in SERVED_ELSEWHERE below rather than papered over.
"""

import io
import os
import re
import sys

SERVER = os.path.join("src", "server.rs")
SPEC = os.path.join("src", "openapi_export.rs")

# Routes the server serves without a `("METHOD", "/path")` arm, with the reason.
# An entry here is a claim that the route IS served; verify before adding one.
SERVED_ELSEWHERE = {
    # `let is_ws = (request.path == "/ws" || request.path == "/api/v1/ws")` — the upgrade
    # is decided before the routing match, so no arm exists.
    "/ws",
    "/api/v1/ws",
}

# Paths in the spec that are templated (`/sessions/{id}`) match prefix-based arms rather
# than literal ones. Mapped to the prefix the server actually checks.
TEMPLATED = {
    "/sessions/{id}": "/sessions/",
}


def served_routes(src):
    """Literal routes from the dispatch match arms."""
    routes = set(m.group(2) for m in re.finditer(r'\("(GET|POST|PUT|DELETE|PATCH)", "([^"]+)"\)', src))
    # Prefix arms: `("DELETE", path) if path.starts_with("/sessions/")`
    prefixes = set(re.findall(r'path\.starts_with\("([^"]+)"\)', src))
    return routes, prefixes


def declared_paths(src):
    """Keys of the `paths` map inside `generate_server_api_spec`."""
    i = src.find("pub fn generate_server_api_spec")
    if i < 0:
        return None
    body = src[i:]
    j = body.find('"paths": {')
    if j < 0:
        return None
    # From the opening brace, take until the matching close.
    start = body.index("{", j + len('"paths":'))
    depth, k = 0, start
    while k < len(body):
        if body[k] == "{":
            depth += 1
        elif body[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return set(re.findall(r'"(/[^"]*)"\s*:', body[start:k]))


def main():
    if not os.path.exists(SERVER) or not os.path.exists(SPEC):
        print("error: run this from the repository root")
        return 2

    routes, prefixes = served_routes(io.open(SERVER, encoding="utf-8").read())
    declared = declared_paths(io.open(SPEC, encoding="utf-8").read())
    if declared is None:
        print("error: could not find the `paths` map — has generate_server_api_spec changed shape?")
        return 2

    routes |= SERVED_ELSEWHERE

    undeclared = sorted(r for r in routes if r not in declared)
    unserved = []
    for d in sorted(declared):
        if d in routes:
            continue
        prefix = TEMPLATED.get(d)
        if prefix and prefix in prefixes:
            continue
        unserved.append(d)

    problems = 0
    if undeclared:
        print("SERVED BUT NOT IN THE SPEC (a client generated from it will lack these):")
        for r in undeclared:
            print("   ", r)
        print()
        problems += len(undeclared)
    if unserved:
        print("IN THE SPEC BUT NOT SERVED (a client will call these and get 404):")
        for d in unserved:
            print("   ", d)
        print()
        problems += len(unserved)

    if problems:
        print("%d mismatch(es). The spec at /openapi.json is a contract: it is what a" % problems)
        print("third party generates their client from.")
        return 1

    print("OK - %d routes, all declared and all served." % len(routes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
