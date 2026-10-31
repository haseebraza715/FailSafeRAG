# TLS "bad record MAC" drops on large requests

**Kind: engineering check, 2026-10-31. No model was called and nothing was billed.** Every request in this check
was an unauthenticated POST to `https://api.openai.com/v1/chat/completions`, which the server answers with 401.

## The problem

In the four image runs on `ohr_dev_v1`, 11 of 280 requests ended with
`ReadError: [SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac`, outcome unknown. None of the 394
text-only requests in the six text runs did. The runner never resends an unknown outcome, so each drop became an
`execution_failed` question scored 0. That biases a comparison against the image condition.

The alert means the server received a TLS record whose integrity check failed and closed the connection. The
records of the 11 drops show:

- every drop came 0 to 1 s after dispatch, much faster than a normal reply (median about 1 s);
- none was the first request of an invocation (5 first requests succeeded), so connection reuse was a suspect;
- the dropped bodies were not unusually large among image requests (between the 41st and 80th percentile).

## What was tested

`probe.py` sends the same kind of body as the image runs, through the same client the runs use
(`openai.DefaultHttpxClient(follow_redirects=False, trust_env=False)`, httpx 0.28.1, OpenSSL 3.5.7, HTTP/1.1; the
`h2` package is not installed, so HTTP/2 is never used). Each large body holds one ohr_dev_v1 page JPEG in base64
with `detail: high`, 103 KB to 1.48 MB, mean 503 KB. The small body is 1.3 KB of text. `curl_probe.sh` repeats the
large-body test with macOS curl 8.7.1 on LibreSSL 3.3.6, a different TLS library in a separate process.

| Arm | Client | Connection | Body | Requests | Bad record MAC | Other |
| --- | --- | --- | --- | ---: | ---: | --- |
| `reuse_big` | Python, OpenSSL | one connection, reused | large | 200 | 6 | one 502 |
| `fresh_big` | Python, OpenSSL | new connection per request | large | 200 | 3 | |
| `reuse_small` | Python, OpenSSL | one connection, reused | 1.3 KB | 200 | 0 | |
| curl | curl, LibreSSL | new process per request | large | 200 | 1 | one 502, one 520 |

Raw results: `probe1.jsonl` (Python arms, one line per request, with the error, body size and time to failure) and
`curl1.txt` (curl exit code and HTTP status per request). The Python arms ran from 22:04 to 22:07 UTC, the curl arm
right after.

## What the results show

- **Body size matters.** Large bodies dropped 9 of 400 times (2.3%), small ones 0 of 200. This matches 11 of 280
  image requests against 0 of 394 text requests in the real runs.
- **Connection reuse is not the cause.** Fresh connections dropped too (3 of 200, against 6 of 200 reused). The
  difference is within chance.
- **The client library is not the cause.** curl on LibreSSL, a separate TLS implementation, got the same alert.
- **HTTP/2 is not involved.** Every request used HTTP/1.1.
- **The image detail setting and the base64 content cannot be the cause.** The alert is raised at the TLS layer,
  before the server parses the JSON body. The same bodies that dropped once succeeded on other attempts.
- Each drop came 23 to 130 ms after the upload started, part way through the body.

**Interpretation.** Bytes of the encrypted upload are being corrupted between this machine and the server, below
TLS, on the local network path. TCP checksums are weak enough to let rare corruption through, and a larger upload
gives it more chances. This cannot be fixed in the runner. A wired connection or a different network might remove
it. That was not tested.

**What it means for billing.** The server rejected the TLS record before the request body was complete, so the model
most likely never ran and nothing was billed. This cannot be confirmed from the client side. The runs keep counting
each dropped attempt at its upper bound.

## Consequence

Shrinking the images would lower the drop rate but would change the image settings of the pre-registered design.
Instead, the repair analysis plan adopts the resend rule `one-resend-connection-lost-v1`
(`docs/research/repair-analysis-plan.md`, amendment 1): an attempt that ends `connection_lost` with an unknown
outcome may be sent once more, its first attempt stays counted at its upper bound, and blocked questions are reported
per condition.

## Files

`probe.py` was reformatted for the linter after the run (imports split, two one-line statements split, the unused
`httpx` import removed). Its behaviour is unchanged. The version that ran had SHA-256
`4e27577e3affed904fa7780bdbc76b0e22e60575565897880e66e8ed81b8a8fc`.
