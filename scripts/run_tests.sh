#!/bin/bash
python3 -m pytest --cov-config .coveragerc --cov-report xml:coverage.xml --cov-report term --cov=. -v --color=yes -m "not expensive"