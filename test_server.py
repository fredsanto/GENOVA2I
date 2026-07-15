"""test_server.py — integration tests for ServerQwen."""
import subprocess
import sys
import time
import os
import json
import http.client

SERVER_PORT = 8013
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))


def start_server():
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'server_qwen:app',
         '--host', '0.0.0.0', '--port', str(SERVER_PORT),
         '--log-level', 'error'],
        cwd=SERVER_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_ready(timeout=30):
    import urllib.request
    for i in range(timeout):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f'http://localhost:{SERVER_PORT}/health', timeout=2) as r:
                if r.status == 200:
                    print(f'  Server ready after {i+1}s')
                    return True
        except Exception:
            pass
    return False


def request(method, path, **kwargs):
    import urllib.request
    import urllib.parse
    url = f'http://localhost:{SERVER_PORT}{path}'
    data = kwargs.get('data')
    headers = kwargs.get('headers', {})
    if data and isinstance(data, str):
        data = data.encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=kwargs.get('timeout', 10)) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        return 0, str(e)


def multipart_request(csv_content, phenotype, filename='test.csv'):
    boundary = '----TestBoundary123'
    body = []
    body.append(f'--{boundary}')
    body.append(f'Content-Disposition: form-data; name="csv_file"; filename="{filename}"')
    body.append('Content-Type: text/csv')
    body.append('')
    body.append(csv_content)
    body.append(f'--{boundary}')
    body.append('Content-Disposition: form-data; name="patient_report"')
    body.append('')
    body.append(phenotype)
    body.append(f'--{boundary}--')
    data = '\r\n'.join(body)
    return data.encode(), {
        'Content-Type': f'multipart/form-data; boundary={boundary}',
    }


def test_sse_stream(job_id):
    conn = http.client.HTTPConnection('localhost', SERVER_PORT, timeout=15)
    conn.request('GET', f'/stream/{job_id}')
    resp = conn.getresponse()
    events = []
    for line in resp:
        if line.startswith(b'data: '):
            ev = json.loads(line[6:])
            events.append(ev)
            if ev.get('type') in ('result', 'error'):
                break
    conn.close()
    return resp.status, resp.getheader('Content-Type'), events


def main():
    import urllib.error

    print('Starting server...')
    proc = start_server()

    if not wait_ready():
        print('FAIL Server did not start')
        proc.terminate()
        sys.exit(1)

    passed = 0
    failed = 0

    def check(name, ok):
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f'  PASS {name}')
        else:
            failed += 1
            print(f'  FAIL {name}')

    # 1. Health
    status, body = request('GET', '/health')
    data = json.loads(body)
    check('Health 200', status == 200)
    check('Health status ok', data['status'] == 'ok')
    check('Backend reported', data['backend'] == 'proxy')

    # 2. Web UI
    status, body = request('GET', '/')
    check('Index 200', status == 200)
    check('Has LIVE badge', 'LIVE' in body)
    check('Has submit button', 'submitBtn' in body)
    check('Has file input', 'csvFile' in body)
    check('Has phenotype input', 'phenotype' in body)

    # 3. Analyze
    csv = ('Variant,Chromosome,Position,RS_ID,Ref_seq,Var_seq,Type,HGVS,Zygosity,Gene\n'
           'chr6:100896130 T>C,chr6,100896130,NA,T,C,SNV,NM_005068.3:c.744-2A>G p.?,Heterozygous,SIM1\n')
    data, headers = multipart_request(csv, 'Obesity, developmental delay', 'sim1.csv')
    status, body = request('POST', '/analyze', data=data, headers=headers)
    check('Analyze 200', status == 200)
    result = json.loads(body)
    jid = result.get('job_id', '')
    check('Has job_id', bool(jid))
    check('Has stream_url', '/stream/' in result.get('stream_url', ''))

    # 4. SSE stream
    sse_status, sse_ct, events = test_sse_stream(jid)
    check('SSE status 200', sse_status == 200)
    check('SSE content-type', sse_ct and 'text/event-stream' in sse_ct)
    check('SSE has events', len(events) >= 1)
    check('SSE final is terminal', len(events) > 0 and events[-1]['type'] in ('result', 'error'))
    if events:
        types = [e["type"] for e in events]
        print(f'  SSE event types: {types}')

    # 5. Result endpoint
    status, body = request('GET', f'/result/{jid}')
    check('Result 200', status == 200)
    data = json.loads(body)
    check('Result has status', data.get('status') in ('running', 'done'))

    # 6. Empty CSV rejected
    data, headers = multipart_request('', 'test', 'empty.csv')
    status, body = request('POST', '/analyze', data=data, headers=headers)
    check('Empty CSV 400', status == 400)

    # 7. Invalid job ID
    status, body = request('GET', '/stream/nonexistent')
    check('Bad job 404', status == 404)

    # 8. Multiple concurrent jobs
    jids = set()
    for i in range(5):
        data, headers = multipart_request(csv, f'phenotype {i}', f'test{i}.csv')
        status, body = request('POST', '/analyze', data=data, headers=headers)
        if status == 200:
            jids.add(json.loads(body)['job_id'])
    check('5 unique jobs', len(jids) == 5)

    # 9. All streams work
    all_ok = True
    for j in list(jids)[:3]:
        s, ct, evs = test_sse_stream(j)
        if s != 200 or len(evs) == 0:
            all_ok = False
    check('All concurrent streams work', all_ok)

    print(f'\n{"="*40}')
    print(f'RESULTS: {passed} passed, {failed} failed')
    print(f'{"="*40}')

    proc.terminate()
    proc.wait()
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
