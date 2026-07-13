import cdsapi
import os
from dotenv import load_dotenv
from agenticeoecosystem.definitions import ROOT_DIR
# print("ROOT_DIR:", ROOT_DIR)
env_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(env_path)
url = os.getenv("CDS_URL")
key = os.getenv("CDS_KEY")
# os.environ['CDSAPI_RC'] = './.cdsapirc'  # or any path, e.g. os.path.join(os.path.dirname(__file__), '.cdsapirc')

client = cdsapi.Client(url=url, key=key)

dataset = 'reanalysis-era5-pressure-levels'
request = {
  'product_type': ['reanalysis'],
  'variable': ['geopotential'],
  'year': ['2024'],
  'month': ['03'],
  'day': ['01'],
  'time': ['13:00'],
  'pressure_level': ['1000'],
  'data_format': 'grib',
}
target = 'download.grib'

client.retrieve(dataset, request, target)