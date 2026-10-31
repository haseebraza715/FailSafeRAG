#!/bin/sh
# The curl arm of the TLS check: 200 unauthenticated POSTs of large JSON bodies over macOS curl (LibreSSL),
# one process and one fresh connection per request. The server answers 401; nothing is billed.
# body0.json, body1.json, body2.json: chat-completions bodies, each with one ohr_dev_v1 page JPEG in base64
# (583,809, 1,113,053 and 1,113,053 bytes). They are not kept; any page JPEG of that size reproduces the arm.
for i in $(seq 1 200); do
  b=body$((i % 3)).json
  out=$(curl -sS --http1.1 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
    --data-binary @$b https://api.openai.com/v1/chat/completions 2>&1)
  echo "$i $? $b $out"
done > curl1.txt
