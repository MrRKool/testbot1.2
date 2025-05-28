import os
import subprocess
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import colorama
from colorama import Fore, Style
import tempfile

# Initialiseer colorama voor kleur in de terminal
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Maak logs directory als die niet bestaat
os.makedirs("logs", exist_ok=True)
handler = logging.FileHandler("logs/health_analyzer.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def print_status(message: str, status: str = "info"):
    """Print een bericht met kleur en status."""
    colors = {
        "info": Fore.BLUE,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED
    }
    color = colors.get(status, Fore.WHITE)
    print(f"{color}[{status.upper()}] {message}{Style.RESET_ALL}")

def get_python_files() -> List[str]:
    """Get all Python files in the project, excluding venv and cache directories."""
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip virtual environment and cache directories
        if 'venv' in dirs:
            dirs.remove('venv')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def run_pylint(file_path: str) -> Dict[str, Any]:
    """Run pylint on a single file and return the results."""
    try:
        result = subprocess.run(
            ['pylint', '--rcfile=.pylintrc', '--output-format=text', file_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout per file
        )
        
        return {
            'file': file_path,
            'exit_code': result.returncode,
            'output': result.stdout,
            'error': result.stderr
        }
    except subprocess.TimeoutExpired:
        logger.error(f"Pylint timed out for {file_path}")
        return {
            'file': file_path,
            'exit_code': -1,
            'output': '',
            'error': 'Timeout after 30 seconds'
        }
    except Exception as e:
        logger.error(f"Error running pylint on {file_path}: {str(e)}")
        return {
            'file': file_path,
            'exit_code': -1,
            'output': '',
            'error': str(e)
        }

def main():
    """Main function to run the health analyzer."""
    logger.info("Starting health analyzer")
    
    # Get all Python files
    python_files = get_python_files()
    logger.info(f"Found {len(python_files)} Python files to analyze")
    
    # Create a temporary file for the report
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as report_file:
        report_file.write("Health Analysis Report\n")
        report_file.write("=====================\n\n")
        
        total_files = len(python_files)
        analyzed_files = 0
        failed_files = 0
        
        for file_path in python_files:
            logger.info(f"Analyzing {file_path}")
            result = run_pylint(file_path)
            
            report_file.write(f"\nFile: {file_path}\n")
            report_file.write("-" * (len(file_path) + 6) + "\n")
            
            if result['exit_code'] == 0:
                report_file.write("Status: PASSED\n")
                analyzed_files += 1
            else:
                report_file.write("Status: FAILED\n")
                failed_files += 1
                if result['output']:
                    report_file.write("\nIssues found:\n")
                    report_file.write(result['output'])
                if result['error']:
                    report_file.write("\nErrors:\n")
                    report_file.write(result['error'])
            
            report_file.write("\n")
        
        # Write summary
        report_file.write("\nSummary\n")
        report_file.write("=======\n")
        report_file.write(f"Total files analyzed: {total_files}\n")
        report_file.write(f"Successfully analyzed: {analyzed_files}\n")
        report_file.write(f"Failed to analyze: {failed_files}\n")
        
        if total_files > 0:
            success_rate = (analyzed_files / total_files) * 100
            report_file.write(f"Success rate: {success_rate:.1f}%\n")
    
    logger.info(f"Analysis complete. Report saved to {report_file.name}")
    print(f"\nAnalysis complete. Report saved to {report_file.name}")

if __name__ == "__main__":
    main() 