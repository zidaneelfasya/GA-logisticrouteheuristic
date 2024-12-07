import pandas as pd

file_path = "data_route.xlsx"
xls = pd.ExcelFile(file_path)
print(xls.sheet_names)