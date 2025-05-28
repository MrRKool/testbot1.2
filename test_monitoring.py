import sys
import os
from datetime import datetime, timedelta
from utils.monitoring.dashboard import Dashboard, DashboardConfig
from utils.monitoring.daily_reporter import DailyReporter, ReportConfig, ReportType
from utils.telegram_alerts import TelegramAlert
from utils.env_loader import check_environment
from utils.logger import Logger

def test_monitoring():
    print("Testing monitoring functionality...")
    
    # Initialize logger
    logger = Logger()
    log = logger.get_logger(__name__)
    
    # Test dashboard
    print("\nTesting Dashboard...")
    try:
        dashboard = Dashboard()
        log.info("Dashboard initialized successfully")
        print("✅ Dashboard initialized successfully")
        
        # Test reporter
        print("\nTesting DailyReporter...")
        reporter = DailyReporter()
        log.info("DailyReporter initialized successfully")
        print("✅ DailyReporter initialized successfully")
        
        # Generate a test report
        print("\nGenerating test report...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        report_path = reporter.generate_report(
            report_type=ReportType.DAILY,
            start_date=start_date,
            end_date=end_date
        )
        if report_path:
            log.info(f"Report generated successfully: {report_path}")
            print(f"✅ Report generated successfully: {report_path}")
        else:
            log.info("No report generated (this is normal if there are no trades)")
            print("⚠️ No report generated (this is normal if there are no trades)")
            
    except Exception as e:
        log.error(f"Error during testing: {e}")
        print(f"❌ Error during testing: {e}")
        return False
        
    print("\n✅ All monitoring tests completed successfully")
    return True

if __name__ == "__main__":
    test_monitoring() 