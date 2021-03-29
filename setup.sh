#!/usr/bin/env bash

source env/bin/activate
pip install -r requirements.txt

export QT_SCALE_FACTOR=2.0
python start.py
