from io import BytesIO
import os
from os.path import dirname, join, realpath, getsize
import logging
from env_loader import load_env
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS


import sys
# In order to run this script directly, you need to add the parent directory to the sys.path
# Or you need to run this script in the parent directory using the command: python -m server.server
sys.path.append(dirname(realpath('.')))
from common.config_logging import init_logging
logger = init_logging('server.log')

# Load the common configurations from .env file and environment variables
env = load_env("./", export_to_env=True)
SERVER_PORT = env.get("SERVER_PORT", 8080)
SERVER_LISTEN_IP = env.get("SERVER_LISTEN_IP", "0.0.0.0")
logging.info(f"Server started on {SERVER_LISTEN_IP}:{SERVER_PORT}")

# record the startup datetime of the server
server_startup_datetime = datetime.now()
logging.info(f"Server startup datetime: {server_startup_datetime}")


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    # format the current datetime
    return f"<p>Server started since {server_startup_datetime}</p>"


@app.route("/get", methods=["GET"])
def get():
    # get the filename and size query parameters
    filename = request.args.get("filename", "file.bin")
    size = request.args.get("size", "10K")
    # parse the size like 1024, 10K and 10M to int
    size = (
        int(size[:-1]) * 1024
        if size[-1] == "K"
        else int(size[:-1]) * 1024 * 1024
        if size[-1] == "M"
        else int(size)
    )
    # generated sized arbitrary data
    data = os.urandom(size)
    # return the data
    return send_file(BytesIO(data), download_name=filename, as_attachment=True)


@app.route("/put", methods=["PUT"])
def put():
    file = request.files["file"]
    # save the uploaded file to a temporary file
    temp_file_path = join(dirname(realpath(__file__)), "__tmp__")
    file.save(temp_file_path)
    file_upload_size = getsize(temp_file_path)
    logging.info(
        f"processed uploaded file: {file.filename}, saved to {temp_file_path}, size of {file_upload_size}"
    )
    # simple return success json response
    return jsonify(
        {
            "code": 0,
            "message": "success",
            "data": f"uploaded file size: {file_upload_size}",
        }
    )


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


if __name__ == "__main__":
    app.run(host=SERVER_LISTEN_IP, port=SERVER_PORT)
