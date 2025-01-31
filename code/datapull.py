#%%
import pandas as pd
from credentials import *
from paths import *
from pathlib import Path
from ipumspy import IpumsApiClient, MicrodataExtract, readers, ddi
IPUMS_API_KEY = apikeys['robipumskey']
ipums = IpumsApiClient(IPUMS_API_KEY)
datadir = Path(ProjectPaths['data'])

# %%
CpsYears = list(range(1990, 2008))
CpsSampleIds = ['cps' + str(year) + '_03s' for year in CpsYears] # https://cps.ipums.org/cps-action/samples/sample_ids
CpsVars = ['STATEFIP','MIGSTA1', 'MIGRATE1', 'INCTOT', 'INCWAGE', 'IND90LY', 'OCC90LY',
           'WKSWORK1', 'UHRSWORKLY', 'EDUC', 'AGE']
extract = MicrodataExtract(collection = 'cps', 
                           description = 'CPS sample for MigrationBorrowing Paper',
                           samples = CpsSampleIds,
                           variables = CpsVars)
ipums.submit_extract(extract)
print(f"Extract submitted with id {extract.extract_id}")
ipums.wait_for_extract(extract)
ipums.download_extract(extract, download_dir = datadir)

# %%
# Get the DDI
ddi_file = list(datadir.glob("*.xml"))[0]
ddi = readers.read_ipums_ddi(ddi_file)

# Get and save data
ipums_df = readers.read_microdata(ddi, datadir / ddi.file_description.filename)
ipums_df.to_csv(ProjectPaths['data'] + '/CpsPull.csv', index=False)
# %%
