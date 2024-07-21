#!/usr/bin/python3.6
####################
# submit.py
####################
import os
import htcondor
import subprocess

def main():

    print(htcondor.version())

    os.makedirs("out", exist_ok=True)
    os.makedirs("result", exist_ok=True)

    subprocess.run(["tar", "-czf", ".venv.tar.gz", ".venv"])

    schedd = htcondor.Schedd()

    sub = htcondor.Submit({
            "executable": "job.sh",
            "output": "out/$(Process).out",
            "log": "out/$(Process).log",
            "error": "out/$(Process).err",
            "should_transfer_files": "YES",
            "transfer_input_files": "job.py, .venv.tar.gz, libs",
            "transfer_output_files": "result",
        })

    mds = [10, 25, 50, 100, 200, 400]
    sdps = [.00, .05, .10, .25, .50]
    rs = [0.01, 0.04, 0.1, 0.4, 1., 2.]
    bs = [0, 1, 2, 3, 5]

    i = 1
    for md in mds:
        for sdp in sdps:
            with schedd.transaction() as txn:
                for r in rs:
                    for b in bs:
                        sub["arguments"] = f"--md {md} --sdp {sdp} -r {r} -b {b} --rng_seed {i}"
                        print(sub["arguments"])
                        sub.queue(txn)
                        i += 1

if __name__ == "__main__":
    main()
