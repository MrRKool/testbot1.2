#!/usr/bin/env python3
"""
Test script for the trading bot.
"""

import argparse
import logging
import os
import sys
import unittest
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

def run_unit_tests():
    """Run unit tests."""
    try:
        import pytest

        # Run pytest with coverage
        pytest.main(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "tests/unit",
            ]
        )
        logger.info("Unit tests completed successfully")
    except Exception as e:
        logger.error(f"Failed to run unit tests: {e}")
        sys.exit(1)

def run_integration_tests():
    """Run integration tests."""
    try:
        import pytest

        # Run pytest with coverage
        pytest.main(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "tests/integration",
            ]
        )
        logger.info("Integration tests completed successfully")
    except Exception as e:
        logger.error(f"Failed to run integration tests: {e}")
        sys.exit(1)

def run_performance_tests():
    """Run performance tests."""
    try:
        import pytest

        # Run pytest with coverage
        pytest.main(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "tests/performance",
            ]
        )
        logger.info("Performance tests completed successfully")
    except Exception as e:
        logger.error(f"Failed to run performance tests: {e}")
        sys.exit(1)

def run_stress_tests():
    """Run stress tests."""
    try:
        import pytest

        # Run pytest with coverage
        pytest.main(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "tests/stress",
            ]
        )
        logger.info("Stress tests completed successfully")
    except Exception as e:
        logger.error(f"Failed to run stress tests: {e}")
        sys.exit(1)

def run_security_tests():
    """Run security tests."""
    try:
        import pytest

        # Run pytest with coverage
        pytest.main(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "tests/security",
            ]
        )
        logger.info("Security tests completed successfully")
    except Exception as e:
        logger.error(f"Failed to run security tests: {e}")
        sys.exit(1)

def run_all_tests():
    """Run all tests."""
    try:
        import pytest

        # Run pytest with coverage
        pytest.main(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "tests",
            ]
        )
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Failed to run all tests: {e}")
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run trading bot tests")
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run stress tests",
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run code linting",
    )
    parser.add_argument(
        "--type-check",
        action="store_true",
        help="Run type checking",
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security checks",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests and checks",
    )
    return parser.parse_args()

def main():
    """Main function to run tests."""
    args = parse_args()

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    if args.all:
        run_unit_tests()
        run_integration_tests()
        run_performance_tests()
        run_coverage()
        run_linting()
        run_type_checking()
        run_security_checks()
    else:
        if args.unit:
            run_unit_tests()
        if args.integration:
            run_integration_tests()
        if args.performance:
            run_performance_tests()
        if args.coverage:
            run_coverage()
        if args.lint:
            run_linting()
        if args.type_check:
            run_type_checking()
        if args.security:
            run_security_checks()

    logger.info("All tests completed successfully")

if __name__ == "__main__":
    main() 