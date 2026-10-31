"""Free TLS probe: unauthenticated POSTs to the chat endpoint (401, nothing billed).

usage: probe.py ARM N OUT.jsonl   ARM in reuse_big, fresh_big, reuse_small, reuse_big_nokeepalive
"""
import base64
import glob
import json
import random
import sys
import time

import openai

arm, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
URL = "https://api.openai.com/v1/chat/completions"
pages = sorted(glob.glob("/Users/haseeb/Thesis-Paper/Code/results/analysis/2026-10-31-ohr-dev-v1-terra-failures/pages/*.model.jpg"))
random.seed(1)

def body(big):
    content = [{"type": "text", "text": "probe " * 200}]
    if big:
        data = open(random.choice(pages), "rb").read()
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode(), "detail": "high"}})
    return json.dumps({"model": "gpt-6.1-terra", "messages": [{"role": "user", "content": content}]}).encode()

def client():
    return openai.DefaultHttpxClient(follow_redirects=False, trust_env=False)

big = "big" in arm
c = client()
with open(out, "a") as f:
    for i in range(n):
        if arm.startswith("fresh"):
            c.close()
            c = client()
        b = body(big)
        t = time.monotonic()
        try:
            r = c.post(URL, content=b, headers={"Content-Type": "application/json"})
            res = {"status": r.status_code, "http": r.http_version}
        except Exception as e:
            res = {"error": type(e).__name__, "msg": str(e)[:200]}
        res.update(arm=arm, i=i, bytes=len(b), ms=int((time.monotonic() - t) * 1000), at=time.time())
        f.write(json.dumps(res) + "\n")
        f.flush()
c.close()
