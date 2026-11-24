from app.observability import initialize
initialize()
from app.db import connection
from worker.ingest import process_s3_key

with connection() as conn:
    conn.execute("UPDATE documents SET status='pending' WHERE status IN ('ready','failed','processing')")
    keys = [r['s3_key'] for r in conn.execute(
        "SELECT s3_key FROM documents WHERE status='pending'").fetchall()]

for k in keys:
    try:
        process_s3_key(k)
        print('OK  ', k)
    except Exception as e:
        print('FAIL', k, e)