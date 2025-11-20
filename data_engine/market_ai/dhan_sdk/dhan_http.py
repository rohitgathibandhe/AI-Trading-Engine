"""
    A class to interact with the DhanHQ APIs using HTTP protocol.

    This library provides methods to manage orders, retrieve market data,
    and perform various trading operations through the DhanHQ API.

    :copyright: (c) 2025 by Dhan.
    :license: see LICENSE for details.
"""

import logging
import os
import time
from datetime import date, datetime, time as dtime
from decimal import Decimal
from enum import Enum
from json import dumps as json_dumps, loads as json_loads

import requests
from requests import exceptions as req_exc


class DhanHTTP:
    """Manages API keys, connection context, and HTTP requests"""

    class HttpResponseStatus(Enum):
        """Constants for HTTP Status Codes"""
        SUCCESS = 'success'
        FAILURE = 'failure'

    class HttpMethods(Enum):
        """Constants for HTTP Requests"""
        GET = 'GET'
        POST = 'POST'
        PUT = 'PUT'
        DELETE = 'DELETE'

    HTTP_DEFAULT_TIME_OUT = 60
    API_BASE_URL = 'https://api.dhan.co/v2'

    def __init__(self, client_id, access_token, disable_ssl=False, pool=None):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = DhanHTTP.API_BASE_URL
        self.timeout = int(os.getenv("DHAN_HTTP_TIMEOUT", DhanHTTP.HTTP_DEFAULT_TIME_OUT))
        self.max_retries = max(1, int(os.getenv("DHAN_HTTP_RETRIES", 2)))
        self.header = {
            'access-token': self.access_token,
            'client-id': self.client_id,
            'Content-type': 'application/json',
            'Accept': 'application/json'
        }
        self.disable_ssl = disable_ssl
        self.session = requests.Session()
        if pool:
            reqadapter = requests.adapters.HTTPAdapter(**pool)
            self.session.mount("https://", reqadapter)

    @staticmethod
    def _sanitize_payload(payload):
        """
        Recursively convert unsupported types (date/time/Decimal, etc.) into JSON-friendly values.
        """
        def convert(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            if isinstance(value, dtime):
                return value.isoformat()
            if isinstance(value, Decimal):
                return float(value)
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [convert(v) for v in value]
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
                    pass
            return str(value)

        return convert(payload)

    def _send_request(self, method, endpoint, payload=None):
        url = self.base_url + endpoint
        if payload:
            payload = self._sanitize_payload(payload)
            if isinstance(payload, dict):
                payload["dhanClientId"] = self.client_id
            else:
                payload = {
                    "dhanClientId": self.client_id,
                    "data": payload,
                }
            payload = json_dumps(payload)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = getattr(self.session, method.value.lower())(
                    url,
                    data=payload,
                    headers=self.header,
                    timeout=self.timeout,
                )
                return self._parse_response(response)
            except req_exc.ReadTimeout as e:
                last_error = e
                logging.warning(
                    'Timeout in DhanHQConnection.%s (attempt %d/%d, timeout=%ss)',
                    method.value.upper(),
                    attempt,
                    self.max_retries,
                    self.timeout,
                )
            except req_exc.ConnectionError as e:
                last_error = e
                logging.warning(
                    'Connection error in DhanHQConnection.%s (attempt %d/%d): %s',
                    method.value.upper(),
                    attempt,
                    self.max_retries,
                    e,
                )
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = requests.Session()
            except Exception as e:  # noqa: BLE001
                last_error = e
                break
            if attempt < self.max_retries:
                sleep_for = min(1.0 * attempt, 3.0)
                time.sleep(sleep_for)
        if last_error:
            logging.error('Exception in DhanHQConnection.%s: %s', method.value.upper(), last_error)
        return {
            'status': DhanHTTP.HttpResponseStatus.FAILURE.value,
            'remarks': str(last_error) if last_error else 'unknown error',
            'data': '',
        }

    def _parse_response(self, response):
        """
        Parse the API's string response to return a JSON as dict.

        Args:
            response (requests.Response): The response object from the API.

        Returns:
            dict: Parsed response containing status, remarks, and data.
        """
        try:
            status = DhanHTTP.HttpResponseStatus.FAILURE.value
            remarks = ''
            data = ''
            json_response = json_loads(response.content)
            if (response.status_code >= 200) and (response.status_code <= 299):
                status = DhanHTTP.HttpResponseStatus.SUCCESS.value
                data = json_response
            else:
                remarks = {
                    'error_code': (json_response.get('errorCode')),
                    'error_type': (json_response.get('errorType')),
                    'error_message': (json_response.get('errorMessage'))
                }
        except Exception as e:
            logging.warning('Exception found in dhanhq>>find_error_code: %s', e)
            status = DhanHTTP.HttpResponseStatus.FAILURE.value
            remarks = str(e)
        return {
            'status': status,
            'remarks': remarks,
            'data': data,
        }

    def get(self, endpoint):
        """
        Do HTTP-GET request to Dhan Endpoint.

        Args:
            endpoint (str): The endpoint ignoring the base URL.

        Returns:
        dict: The response in dict format.
        """
        return self._send_request(DhanHTTP.HttpMethods.GET, endpoint)

    def post(self, endpoint, payload):
        """
        Do HTTP-POST request to Dhan Endpoint.

        Args:
            endpoint (str): The endpoint ignoring the base URL.
            payload (dict): The payload dict contains the data that needs to be sent to the server.

        Returns:
        dict: The response in dict format.
        """
        return self._send_request(DhanHTTP.HttpMethods.POST, endpoint, payload)

    def put(self, endpoint, payload):
        """
        Do HTTP-PUT request to Dhan Endpoint.

        Args:
            endpoint (str): The endpoint ignoring the base URL.
            payload (dict): The payload dict contains the data that needs to be sent to the server.

        Returns:
        dict: The response in dict format.
        """
        return self._send_request(DhanHTTP.HttpMethods.PUT, endpoint, payload)

    def delete(self, endpoint):
        """
        Do HTTP-DELETE request to Dhan Endpoint.

        Args:
            endpoint (str): The endpoint ignoring the base URL.

        Returns:
        dict: The response in dict format.
        """
        return self._send_request(DhanHTTP.HttpMethods.DELETE, endpoint)
