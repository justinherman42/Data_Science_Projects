import io
import sys
import requests
import pandas as pd

def get_url_csv(url):
    """inputs html link
    outputs pandas df of csv """
    try:
        my_content = requests.get(url).content
        data = pd.read_csv(io.StringIO(my_content.decode('utf-8')))
    except requests.exceptions.RequestException as e:
        print(e)
        sys.exit(1)
    return (data)

# Load in datasets


train = get_url_csv('https://raw.githubusercontent.com/justinherman42/data_606/master/train.csv')
if len(train) > 0:
    print(train.shape)
else:
    print("Train dataframe did not load,Github must have changed address")
test = get_url_csv('https://raw.githubusercontent.com/justinherman42/data_606/master/test.csv')
if len(test) > 0:
    print(test.shape)
else:
    print("Test dataframe did not load.  Github must have changed address")
if len(test) > 0 and len(train) > 0:
    print("\n Function loaded and train test dataframe created")
