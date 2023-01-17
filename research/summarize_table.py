""" summarize table based on exports/table.json  """

import json

with open("exports/table.json") as f:
    rows = json.load(f)


for row in rows:
    print(f'{row["model"]};{row["rer"]*100}')
