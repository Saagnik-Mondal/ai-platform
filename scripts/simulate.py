"""
ML Sentinel — Demo Simulation Script
Sends realistic prediction requests to test the full pipeline.
Run this after docker compose up to populate dashboards and trigger drift detection.

Usage:
    python simulate.py --mode normal       # Send normal transactions
    python simulate.py --mode drift        # Send drifted data to trigger retraining
    python simulate.py --mode mixed        # Mix of normal + drifted
    python simulate.py --mode load         # Load test with concurrent requests
"""

import argparse
import json
import time
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

API_URL = "http://localhost:8000"

# Sample normal transaction (non-fraud)
NORMAL_TEMPLATE = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
}

# Sample fraud transaction
FRAUD_TEMPLATE = {
    "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99,
    "V5": -0.52, "V6": -1.43, "V7": -2.54, "V8": 1.39,
    "V9": -2.77, "V10": -5.22, "V11": 5.21, "V12": -6.97,
    "V13": 0.98, "V14": -8.13, "V15": 0.31, "V16": -5.21,
    "V17": -5.67, "V18": -2.51, "V19": 0.79, "V20": 0.23,
    "V21": 0.54, "V22": 0.05, "V23": -0.30, "V24": -0.07,
    "V25": 0.51, "V26": 0.05, "V27": 1.73, "V28": 0.70,
    "Amount": 1.00
}


def add_noise(template: dict, noise_level: float = 0.1) -> dict:
    """Add random noise to a transaction template."""
    noisy = {}
    for key, value in template.items():
        if key == "Amount":
            noisy[key] = max(0, value + random.gauss(0, value * noise_level * 2))
        else:
            noisy[key] = value + random.gauss(0, abs(value) * noise_level + 0.01)
    return noisy


def create_drifted_transaction() -> dict:
    """Create a transaction with significantly different distribution (to trigger drift)."""
    drifted = {}
    for key in NORMAL_TEMPLATE:
        if key == "Amount":
            # Shift amount distribution significantly
            drifted[key] = max(0, random.gauss(5000, 2000))
        else:
            # Shift feature distributions
            original = NORMAL_TEMPLATE[key]
            drifted[key] = original * random.gauss(3.0, 1.5) + random.gauss(0, 2)
    return drifted


def send_prediction(transaction: dict) -> dict:
    """Send a single prediction request."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=transaction, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def simulate_normal(count: int = 100, delay: float = 0.1):
    """Send normal transactions."""
    print(f"\n{'='*60}")
    print(f"Sending {count} normal transactions...")
    print(f"{'='*60}")

    fraud_count = 0
    for i in range(count):
        # 99.8% normal, 0.2% fraud (realistic ratio)
        if random.random() < 0.002:
            txn = add_noise(FRAUD_TEMPLATE, noise_level=0.2)
        else:
            txn = add_noise(NORMAL_TEMPLATE, noise_level=0.15)

        result = send_prediction(txn)

        if "error" in result:
            print(f"  [{i+1}/{count}] ERROR: {result['error']}")
        else:
            is_fraud = result.get("is_fraud", False)
            if is_fraud:
                fraud_count += 1
            prob = result.get("fraud_probability", 0)
            lat = result.get("latency_ms", 0)
            print(f"  [{i+1}/{count}] {'🚨 FRAUD' if is_fraud else '✅ OK'} "
                  f"| prob: {prob:.4f} | latency: {lat:.1f}ms")

        time.sleep(delay)

    print(f"\nCompleted: {count} predictions, {fraud_count} flagged as fraud")


def simulate_drift(count: int = 200, delay: float = 0.05):
    """Send drifted transactions to trigger drift detection."""
    print(f"\n{'='*60}")
    print(f"Sending {count} DRIFTED transactions to trigger drift detection...")
    print(f"{'='*60}")

    for i in range(count):
        txn = create_drifted_transaction()
        result = send_prediction(txn)

        if "error" in result:
            print(f"  [{i+1}/{count}] ERROR: {result['error']}")
        else:
            prob = result.get("fraud_probability", 0)
            lat = result.get("latency_ms", 0)
            print(f"  [{i+1}/{count}] prob: {prob:.4f} | latency: {lat:.1f}ms")

        time.sleep(delay)

    print(f"\nDrifted data sent! Drift detection should trigger within "
          f"the next scheduled check.")
    print("Check drift status: curl http://localhost:8000/drift/status")


def simulate_mixed(count: int = 300, delay: float = 0.05):
    """Send a mix of normal then drifted transactions."""
    normal_count = count // 2
    drift_count = count - normal_count

    print(f"\nPhase 1: Sending {normal_count} normal transactions...")
    simulate_normal(normal_count, delay)

    print(f"\nPhase 2: Sending {drift_count} drifted transactions...")
    simulate_drift(drift_count, delay)


def simulate_load(count: int = 500, workers: int = 10):
    """Load test with concurrent requests."""
    print(f"\n{'='*60}")
    print(f"Load test: {count} requests with {workers} concurrent workers")
    print(f"{'='*60}")

    transactions = [add_noise(NORMAL_TEMPLATE, 0.15) for _ in range(count)]

    start = time.time()
    results = {"success": 0, "error": 0, "latencies": []}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(send_prediction, txn): txn for txn in transactions}
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                results["error"] += 1
            else:
                results["success"] += 1
                results["latencies"].append(result.get("latency_ms", 0))

    elapsed = time.time() - start
    latencies = results["latencies"]

    print(f"\nResults:")
    print(f"  Total time:   {elapsed:.2f}s")
    print(f"  Throughput:    {count/elapsed:.1f} req/s")
    print(f"  Success:      {results['success']}")
    print(f"  Errors:       {results['error']}")
    if latencies:
        latencies.sort()
        print(f"  Latency p50:  {latencies[len(latencies)//2]:.1f}ms")
        print(f"  Latency p95:  {latencies[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  Latency p99:  {latencies[int(len(latencies)*0.99)]:.1f}ms")


def check_services():
    """Verify all services are running."""
    print("Checking services...")
    services = {
        "API": f"{API_URL}/health",
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana": "http://localhost:3000/api/health",
        "MLflow": "http://localhost:5000/health",
    }
    for name, url in services.items():
        try:
            resp = requests.get(url, timeout=5)
            status = "✅ UP" if resp.status_code == 200 else f"⚠️  {resp.status_code}"
        except Exception:
            status = "❌ DOWN"
        print(f"  {name:12s} {status}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ML Sentinel Demo Simulator")
    parser.add_argument("--mode", choices=["normal", "drift", "mixed", "load", "check"],
                       default="normal", help="Simulation mode")
    parser.add_argument("--count", type=int, default=100, help="Number of requests")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests (seconds)")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers for load test")
    args = parser.parse_args()

    check_services()

    if args.mode == "normal":
        simulate_normal(args.count, args.delay)
    elif args.mode == "drift":
        simulate_drift(args.count, args.delay)
    elif args.mode == "mixed":
        simulate_mixed(args.count, args.delay)
    elif args.mode == "load":
        simulate_load(args.count, args.workers)
    elif args.mode == "check":
        pass  # Just check services

    print("\n📊 View dashboards at: http://localhost:3000")
    print("📡 View MLflow at:     http://localhost:5000")
    print("📄 API docs at:        http://localhost:8000/docs")


if __name__ == "__main__":
    main()
