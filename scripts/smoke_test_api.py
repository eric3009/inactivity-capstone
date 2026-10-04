"""Call a running API (local uvicorn or the Docker container) with a small example customer.

    python scripts/smoke_test_api.py [http://localhost:8000]
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


print("health:", call("/health"))
lines = [
    {"invoice": "500001", "stock_code": "85123", "invoice_date": "2011-01-10T10:00:00", "quantity": 12, "price": 2.5},
    {"invoice": "500002", "stock_code": "84406", "invoice_date": "2011-02-02T11:30:00", "quantity": 6, "price": 4.2},
    {"invoice": "500003", "stock_code": "85123", "invoice_date": "2011-02-20T09:15:00", "quantity": 24, "price": 2.5},
]
print(json.dumps(call("/score/transactions", {"customer_id": "demo-1", "snapshot_date": "2011-03-31",
                                              "transactions": lines}), indent=2))
