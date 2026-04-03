#!/usr/bin/env python3
"""
Comprehensive test suite for Copenhagen - Dynamic IVF Index
Tests with logging and detailed reporting.
"""

import sys
import os
import time
import logging
from datetime import datetime
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from core import CopenhagenIndex

import numpy as np
import faiss

np.random.seed(42)

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration = 0.0
    
    def to_dict(self):
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": self.duration * 1000
        }


def test_basic_insert_search():
    """Test basic insert and search functionality."""
    result = TestResult("basic_insert_search")
    start = time.time()
    
    try:
        logger.info("Testing basic insert and search...")
        
        n, d = 1000, 64
        data = np.random.randn(n, d).astype(np.float32)
        
        idx = CopenhagenIndex(dim=d, n_clusters=32, nprobe=8)
        idx.add(data)
        
        assert idx.n_vectors == n, f"Expected {n} vectors, got {idx.n_vectors}"
        
        query = data[0]
        ids, dists = idx.search(query, k=10)
        
        assert len(ids) == 10, f"Expected 10 results, got {len(ids)}"
        assert ids[0] == 0, f"Expected first result to be 0, got {ids[0]}"
        
        result.passed = True
        result.message = f"Inserted {n} vectors, search returned 10 results correctly"
        logger.info(f"  PASSED: {result.message}")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def test_recall():
    """Test recall against brute force."""
    result = TestResult("recall")
    start = time.time()
    
    try:
        logger.info("Testing recall against brute force...")
        
        n, d = 5000, 64
        n_queries = 50
        
        data = np.random.randn(n, d).astype(np.float32)
        queries = np.random.randn(n_queries, d).astype(np.float32)
        
        idx = CopenhagenIndex(dim=d, n_clusters=32, nprobe=8)
        idx.add(data)
        
        recalls = []
        for q in queries:
            ids, _ = idx.search(q, k=10)
            bf_ids, _ = idx.brute_force_search(q, k=10)
            
            recall = len(set(ids) & set(bf_ids)) / 10
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        
        result.passed = avg_recall >= 0.95
        result.message = f"Average recall@10: {avg_recall:.4f} (target: >= 0.95)"
        
        if result.passed:
            logger.info(f"  PASSED: {result.message}")
        else:
            logger.warning(f"  WARNING: {result.message}")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def test_incremental_insert():
    """Test incremental insert maintains recall."""
    result = TestResult("incremental_insert")
    start = time.time()
    
    try:
        logger.info("Testing incremental insert...")
        
        n1, n2, d = 5000, 5000, 64
        
        data1 = np.random.randn(n1, d).astype(np.float32)
        data2 = np.random.randn(n2, d).astype(np.float32)
        
        idx = CopenhagenIndex(dim=d, n_clusters=64, nprobe=8)
        idx.add(data1)
        idx.add(data2)
        
        assert idx.n_vectors == n1 + n2, f"Expected {n1 + n2}, got {idx.n_vectors}"
        
        query = data1[0]
        ids, _ = idx.search(query, k=10)
        
        recall = len(set(ids) & {0}) / 1.0
        
        result.passed = recall >= 0.9
        result.message = f"After incremental insert of {n2} vectors, recall for original data: {recall:.4f}"
        
        if result.passed:
            logger.info(f"  PASSED: {result.message}")
        else:
            logger.warning(f"  WARNING: {result.message}")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def test_delete():
    """Test delete functionality."""
    result = TestResult("delete")
    start = time.time()
    
    try:
        logger.info("Testing delete functionality...")
        
        d = 64
        idx = CopenhagenIndex(dim=d, n_clusters=16, nprobe=4)
        
        data = np.random.randn(1000, d).astype(np.float32)
        idx.add(data)
        
        assert idx.n_vectors == 1000
        
        spike = np.zeros(d, dtype=np.float32)
        spike[0] = 1e4
        idx.add(spike)
        spike_id = 1000
        
        ids, _ = idx.search(spike, k=5)
        assert spike_id in ids, f"Spike not found before delete"
        
        idx.delete(spike_id)
        
        ids_after, _ = idx.search(spike, k=20)
        assert spike_id not in ids_after, f"Spike still in results after delete"
        
        result.passed = True
        result.message = f"Successfully deleted spike and verified removal from search results"
        logger.info(f"  PASSED: {result.message}")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def test_batch_delete():
    """Test batch delete."""
    result = TestResult("batch_delete")
    start = time.time()
    
    try:
        logger.info("Testing batch delete...")
        
        d = 64
        n = 1000
        
        idx = CopenhagenIndex(dim=d, n_clusters=32, nprobe=4)
        data = np.random.randn(n, d).astype(np.float32)
        idx.add(data)
        
        delete_ids = np.arange(0, 500, 2)
        idx.delete(delete_ids)
        
        assert idx.n_vectors == n - len(delete_ids), f"Expected {n - len(delete_ids)}, got {idx.n_vectors}"
        
        result.passed = True
        result.message = f"Deleted {len(delete_ids)} vectors, {idx.n_vectors} remaining"
        logger.info(f"  PASSED: {result.message}")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def test_nprobe_effect():
    """Test effect of nprobe on recall."""
    result = TestResult("nprobe_effect")
    start = time.time()
    
    try:
        logger.info("Testing nprobe effect on recall...")
        
        n, d = 5000, 64
        n_queries = 50
        
        data = np.random.randn(n, d).astype(np.float32)
        queries = np.random.randn(n_queries, d).astype(np.float32)
        
        recalls = {}
        
        for nprobe in [1, 4, 8, 16]:
            idx = CopenhagenIndex(dim=d, n_clusters=32, nprobe=nprobe)
            idx.add(data)
            
            r = []
            for q in queries:
                ids, _ = idx.search(q, k=10)
                bf_ids, _ = idx.brute_force_search(q, k=10)
                recall = len(set(ids) & set(bf_ids)) / 10
                r.append(recall)
            
            recalls[nprobe] = np.mean(r)
            logger.info(f"  nprobe={nprobe}: recall={recalls[nprobe]:.4f}")
        
        result.passed = recalls[1] < recalls[8] <= recalls[16]
        result.message = f"Recall increases with nprobe: {recalls}"
        
        if result.passed:
            logger.info(f"  PASSED: Recall increases appropriately with nprobe")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def test_different_dimensions():
    """Test with different dimensions."""
    result = TestResult("different_dimensions")
    start = time.time()
    
    try:
        logger.info("Testing different dimensions...")
        
        configs = [(128, 32), (256, 64), (784, 128)]
        
        for d, nc in configs:
            data = np.random.randn(500, d).astype(np.float32)
            idx = CopenhagenIndex(dim=d, n_clusters=nc, nprobe=4)
            idx.add(data)
            
            query = data[0]
            ids, _ = idx.search(query, k=10)
            
            assert len(ids) == 10, f"dim={d}: Expected 10 results, got {len(ids)}"
            logger.info(f"  dim={d}, n_clusters={nc}: OK")
        
        result.passed = True
        result.message = f"Tested {len(configs)} dimension configurations successfully"
        logger.info(f"  PASSED: {result.message}")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def test_large_scale():
    """Test with larger dataset."""
    result = TestResult("large_scale")
    start = time.time()
    
    try:
        logger.info("Testing large scale (10K vectors)...")
        
        n, d = 10000, 128
        
        data = np.random.randn(n, d).astype(np.float32)
        idx = CopenhagenIndex(dim=d, n_clusters=128, nprobe=8)
        
        t0 = time.time()
        idx.add(data)
        add_time = time.time() - t0
        
        query = data[0]
        t0 = time.time()
        ids, dists = idx.search(query, k=10)
        search_time = time.time() - t0
        
        assert len(ids) == 10, f"Expected 10 results, got {len(ids)}"
        assert ids[0] == 0, f"Expected first result to be 0, got {ids[0]}"
        
        result.passed = True
        result.message = f"n={n}, d={d}: add={add_time:.2f}s, search={search_time*1000:.1f}ms"
        
        logger.info(f"  PASSED: {result.message}")
        
    except Exception as e:
        result.message = str(e)
        logger.error(f"  FAILED: {e}")
    
    result.duration = time.time() - start
    return result


def run_all_tests():
    """Run all tests and generate report."""
    logger.info("=" * 60)
    logger.info("COPENHAGEN TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        test_basic_insert_search,
        test_recall,
        test_incremental_insert,
        test_delete,
        test_batch_delete,
        test_nprobe_effect,
        test_different_dimensions,
        test_large_scale,
    ]
    
    results = []
    for test in tests:
        logger.info("")
        r = test()
        results.append(r)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.duration for r in results)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Total time: {total_time:.2f}s")
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "total_duration_s": total_time,
        "results": [r.to_dict() for r in results]
    }
    
    results_file = os.path.join(LOG_DIR, f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Log saved to: {LOG_FILE}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
