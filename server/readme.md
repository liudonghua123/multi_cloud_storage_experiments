# server

The server of adaptive data placement in multi-cloud storage.

The server implementation provide only two endpoints service, `/get` for read data and `/put` for write data.

The server disabled the http cache policy

### How it works

1. `/get` service download arbitrary binary data generated randomly for the specified size and file name.

    | parameter name | type | default value | required? | notes |
    |--------------|-----------|------------|------------|------------|
    | file_name | query string | file.bin | No | the download filename |
    | size      | query string | 10K | No | support 1024, 10K and 10M format |

    You can use the following command alike for testing.

    ```shell
    curl --request GET \
    --url 'http://localhost:8080/get?filename=123.bin&size=10M'
    ```

2. `/put` service upload the specified file, the server only return a successful json respone when the upload process finished.

    | parameter name | type | default value | required? | notes |
    |--------------|-----------|------------|------------|------------|
    | file | multipart/form-data | n.a. | Yes | the file for uploading |

    You can use the following command alike for testing.

    ```shell
    curl --request PUT \
    --url http://localhost:8080/put \
    --header 'content-type: multipart/form-data' \
    --form file=@file
    ```


### What's used in the server implementation

- flask, for simple restful api.
- ~~python-dotenv~~ python-env-loader, for reading .env files in Python from a local environment file.
- flask-cors, for support cors.
- logging, for logging useful log data in stdout and file.

### Some helpful links

- https://github.com/pallets/flask
- https://flask.palletsprojects.com/en/2.2.x/
- https://www.twilio.com/blog/how-run-flask-application
- https://flask-cors.readthedocs.io/en/latest/
- https://www.delftstack.com/howto/python/python-env-file/
- https://github.com/rafaljusiak/python-env-loader
- https://realpython.com/python-logging/
- https://www.geeksforgeeks.org/python-datetime-module/
- https://thewebdev.info/2022/04/03/how-to-disable-caching-in-python-flask/

