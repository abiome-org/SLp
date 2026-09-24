import hashlib, json, re, sys, time, urllib.parse
import http.cookiejar, urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [("User-Agent", UA)]

def solve(url):
    r = op.open(url); body = r.read().decode("utf-8", "replace")
    m = re.search(r'id="anubis_challenge" type="application/json">(.*?)\n</script>', body, re.S)
    if not m:
        return r, body
    ch = json.loads(m.group(1))
    data = ch["challenge"]["randomData"]; diff = ch["rules"]["difficulty"]; cid = ch["challenge"]["id"]
    t0 = time.time(); nonce = 0
    pref = "0" * diff
    while True:
        h = hashlib.sha256((data + str(nonce)).encode()).hexdigest()
        if h.startswith(pref):
            break
        nonce += 1
    q = urllib.parse.urlencode({"id": cid, "response": h, "nonce": nonce,
                                "redir": url, "elapsedTime": int((time.time()-t0)*1000)})
    base = url.split("/", 3)[:3]
    pass_url = "/".join(base) + "/.within.website/x/cmd/anubis/api/pass-challenge?" + q
    r2 = op.open(pass_url)
    return r2, None

def fetch(url, out):
    r, body = solve(url)
    if body is not None and "anubis_challenge" not in body:
        open(out, "wb").write(body.encode()); return
    data = r.read()
    if b"anubis_challenge" in data[:4000]:
        r, _ = solve(url); data = r.read()
    open(out, "wb").write(data)
    print(out, len(data))

if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2])
