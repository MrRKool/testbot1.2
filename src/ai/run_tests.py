import unittest
import logging
from test_ai_system import TestAISystem

def run_tests():
    """Run all AI system tests"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAISystem)
    
    # Run tests
    logger.info("Starting AI system tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    logger.info(f"Tests completed. Results:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    # Print detailed results
    if result.failures:
        logger.error("\nFailures:")
        for failure in result.failures:
            logger.error(f"\n{failure[0]}\n{failure[1]}")
            
    if result.errors:
        logger.error("\nErrors:")
        for error in result.errors:
            logger.error(f"\n{error[0]}\n{error[1]}")
            
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1) 