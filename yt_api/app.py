
import json
import logging
import pandas as pd
from uuid import uuid4
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import select
from sqlalchemy.sql import text

# Logging config
logging.basicConfig(
    #filename='DockProc.log',
    level=logging.INFO, 
    format='[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("espeakng").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



DATABASE_TYPE = "postgresql"
DATABASE_USERNAME = "test"
DATABASE_PASSWORD = "test"
DATABASE_SCHEMA = "youtube"
DATABASE_HOST = "docker.for.mac.localhost"

app = Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI'] = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
#db = SQLAlchemy(app)
#ma = Marshmallow(app)

#url = URL.create(
#    drivername=DATABASE_TYPE,
#    username=DATABASE_USERNAME,
#    password=DATABASE_PASSWORD,
#    host=DATABASE_HOST,
#    database=DATABASE_SCHEMA
#)

#engine = create_engine(url)
#Session = sessionmaker(bind=engine)

#def get_db_session():
#    return Session

#session = get_db_session()


@app.get("/test")
async def root():
    logger.debug("test")
    return {"message": "All good"}



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)