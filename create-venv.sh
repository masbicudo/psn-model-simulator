#!/bin/bash
/usr/bin/python3.6 -m virtualenv -p /usr/bin/python .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy pandas scipy tqdm
