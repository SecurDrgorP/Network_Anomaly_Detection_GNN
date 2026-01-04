#!/bin/bash
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1
pip install nbformat > /dev/null 2>&1
# python3 notebooks/NADGNN.py
python3 main.py
