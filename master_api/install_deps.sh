#!/bin/bash
sudo apt-get update
sudo apt-get install postgresql-devel libpq-dev
pip install sqlalchemy flask uuid uvicorn flask-sqlalchemy flask-marshmallow psycopg2 marshmallow-sqlalchemy flask[async]
