"""
Script for querying the IonQ Cloud API for job information and results, 
including the XRD data used to generate the quantum kernel values.
"""

from scipy.io import loadmat
import requests
import json
import numpy as np

# load x ray diffraction data
data = loadmat("FeGaPd_full_data_220104a.mat")

xrd = data["X"][:, 631:1181]
theta = data["T"].flatten()[631:1181]

# get a list of all the jobs (one for each quantum kernel value)
with open("api_key.txt") as f:
    api_key = f.read()

max_requests = int(600 / 25)
next_id = None

jobs = []

for _ in range(max_requests):
    # query the IonQ Cloud API to get a batch of 25 jobs
    response = requests.get(
        url="https://api.ionq.co/v0.3/jobs",
        headers={"Authorization": f"apiKey {api_key}"},
        params={"next": next_id},
    ).json()

    # filter through each job in the batch
    for job in response["jobs"]:
        if "start" not in job.keys():
            # skip any cancelled jobs
            continue

        if job["start"] > 1724954400:  # jobs started after 2024-08-29 14:00:00
            jobs.append(job)

    # move to the next batch
    next_id = response["next"]

    if next_id is None:
        break

# get the results of each job
def get_result(job):
    # query the IonQ Cloud API to get the results of the jobs
    uuid = job["id"]
    return requests.get(
        url=f"https://api.ionq.co/v0.3/jobs/{uuid}/results",
        headers={"Authorization": f"apiKey {api_key}"},
    ).json()


def ionq_num(job):
    # get the extra numbers at the end of the circuit name...
    # "PetersFeatureMap-###-###"
    return int(job["name"][16:].replace("-", ""))


sorted_jobs = sorted(jobs, key=lambda j: ionq_num(j))
jobs_iter = (lambda: (yield from sorted_jobs))()

# put the jobs and their results in a dictionary with the kernel indices
training_indices = np.array([
    25, 68, 112, 161,
    37, 91, 136, 184,
    77, 85, 128, 224,
    168, 229, 220, 233,
    195, 210, 256, 264,
])

num_points = len(training_indices)

data = {
    "kernel_data": {"num_features": 6 * 25, "num_repetitions": 1},
    "xrd_data": {"xrd": xrd[training_indices].tolist(), "theta": theta.tolist()},
    "kernel_entries": {},
}

for i in range(num_points):
    for j in range(i + 1, num_points):
        print(f"saving k({i}, {j})")

        # add key, value pair to dictionary (json requires String keys)
        job = next(jobs_iter)
        data["kernel_entries"][str((i, j))] = {"job": job, "result": get_result(job)}

path = "kernel_qpu_test_results/"
file_name = "kernel_cpu_test_results_240829.json"

with open(path + file_name, "w") as f:
    f.write(json.dumps(data))
