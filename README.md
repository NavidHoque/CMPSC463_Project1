# CMPSC463_Project1

# Instructions for Running the Stock Data Analysis System

### Step 1: Generate Stock Data CSV
- To get the stock price data, use the **StockPriceGetter.py** script.
- This script will create a CSV file with the stock price data.
- Ensure that the CSV file is generated successfully before moving to the next step.

### Step 2: Use the CSV File in Analysis
- Once the CSV file is created, you need to input the file path of the CSV into the **project1_code.py** script.
- Open the **project1_code.py** script and modify the `file_path` variable to point to the location of your CSV file. Example:
  
  ```python
  file_path = "C:/path/to/your/stock_data.csv"

