import importlib
import sys
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DependencyChecker:
    """Utility class to check and verify all required dependencies."""
    
    REQUIRED_PACKAGES = {
        'core': [
            'pandas',
            'numpy',
            'scikit-learn',
            'scipy'
        ],
        'ml': [
            'xgboost',
            'lightgbm'
        ],
        'data': [
            'yfinance',
            'ta-lib'
        ],
        'optimization': [
            'cvxpy',
            'pulp'
        ],
        'backtesting': [
            'backtrader',
            'vectorbt'
        ]
    }
    
    # Mapping van PyPI package naam naar import-naam
    PACKAGE_IMPORT_MAP = {
        'scikit-learn': 'sklearn',
        'ta-lib': 'talib',
        # Voeg hier meer uitzonderingen toe indien nodig
    }
    
    @classmethod
    def check_dependencies(cls) -> Dict[str, List[str]]:
        """
        Check if all required packages are installed and importable.
        
        Returns:
            Dict[str, List[str]]: Dictionary containing missing packages per category
        """
        missing_packages = {}
        
        for category, packages in cls.REQUIRED_PACKAGES.items():
            category_missing = []
            for package in packages:
                import_name = cls.PACKAGE_IMPORT_MAP.get(package, package)
                try:
                    importlib.import_module(import_name)
                except ImportError as e:
                    category_missing.append(package)
                    logger.error(f"Failed to import {import_name} (package: {package}): {str(e)}")
            
            if category_missing:
                missing_packages[category] = category_missing
        
        return missing_packages
    
    @classmethod
    def verify_installation(cls) -> bool:
        """
        Verify all dependencies and provide helpful error messages.
        
        Returns:
            bool: True if all dependencies are satisfied, False otherwise
        """
        missing = cls.check_dependencies()
        
        if not missing:
            logger.info("All required dependencies are installed correctly.")
            return True
        
        logger.error("Missing required dependencies:")
        for category, packages in missing.items():
            logger.error(f"\n{category.upper()} dependencies:")
            for package in packages:
                logger.error(f"  - {package}")
        
        logger.error("\nTo install missing dependencies, run:")
        logger.error("pip install -r requirements.txt")
        
        return False

def check_dependencies() -> bool:
    """
    Convenience function to check dependencies.
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    return DependencyChecker.verify_installation()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run dependency check
    if not check_dependencies():
        sys.exit(1) 