#!/bin/bash

cd /home/roland_upm/project/myproject/
. venv/bin/activate
export FLASK_APP=srv.py
export FLASK_RUN_PORT=80
flask run -h 0.0.0.0 &
