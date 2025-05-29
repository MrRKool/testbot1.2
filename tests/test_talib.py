import talib
import numpy as np

# Create sample data
close = np.random.random(100)

# Calculate SMA
sma = talib.SMA(close, timeperiod=20)

print("TA-Lib is working correctly!")
print("Sample SMA values:", sma[-5:]) 