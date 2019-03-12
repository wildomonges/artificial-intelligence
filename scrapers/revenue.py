"""
@author: Wildo Monges
email: setwildo31@gmail.com
Job Description: https://www.freelancer.com/projects/data-entry/Data-Scraping-For-List-Building/?w=f
This scraper retrieves data from https://www.inc.com/inc5000/list/2018.
It creates a xlsx file with the following data: Company, Industry, Revenue order by the last one desc
"""
import requests
import json
from openpyxl import Workbook
# Analyzed the main page I found that the data are coming from the next url
page = requests.get('https://www.inc.com/inc5000list/json/inc5000_2018.json')

data = json.loads(page.text)

lines = []
for row in data:
    lines.append(row)

# Sort data from highest revenue to lowest revenue
lines = sorted(lines, key=lambda i: i['revenue'])
lines.reverse()

# Write data to revenue.xlsx
wb = Workbook()
ws = wb.active
# Header
ws.append(['Company', 'Industry', 'Revenue'])
# Iterate over each line of json data
for line in lines:
    ws.append([line['company'], line['industry'], '${0}m'.format(round(line['revenue'] / 1000000, 1))])

wb.save('revenue.xlsx')
