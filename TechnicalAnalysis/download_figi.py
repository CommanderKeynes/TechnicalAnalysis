

import json
import urllib.request
import urllib.parse
import pandas as pd


openfigi_apikey = ''  # Put API Key here


def map_jobs(jobs) -> dict:

    handler = urllib.request.HTTPHandler()
    opener = urllib.request.build_opener(handler)
    openfigi_url = 'https://api.openfigi.com/v3/filter'
    request = urllib.request.Request(openfigi_url, data=bytes(json.dumps(jobs), encoding='utf-8'))
    request.add_header('Content-Type', 'application/json')
    if openfigi_apikey:
        request.add_header('X-OPENFIGI-APIKEY', openfigi_apikey)
    request.get_method = lambda: 'POST'

    connection = opener.open(request)

    return json.loads(connection.read().decode('utf-8'))


def job_results_handler(jobs, job_results) -> None:

    '''
    Handle the `map_jobs` results.  See `map_jobs` definition for more info.
    Parameters
    ----------
    jobs : list(dict)
        The original list of mapping jobs to perform.
    job_results : list(dict)
        The results of the mapping job.
    Returns
    -------
        None
    '''

    print(job_results.keys)()


def main() -> None:

    job = {'exchCode': 'US', }

    job_results = map_jobs(job)
    df = pd.DataFrame(job_results['data'])
    df.to_csv('figi_data.csv', index=False, mode='a')

    while 'next' in job_results.keys():
        job['start'] = job_results['next']
        job_results = map_jobs(job)
        df = pd.DataFrame(job_results['data'])
        df.to_csv('figi_data.csv', index=False, mode='a')


if __name__ == '__main__':
    main()
