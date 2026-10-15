"""Tests for KernelManager (Jupyter kernels REST API)."""

from unittest.mock import Mock, patch

import pytest
import requests

from pyrun_jupyter.exceptions import ConnectionError
from pyrun_jupyter.kernel import KernelManager


class TestKernelManagerAuth:
    """Test how the token is sent to the server."""

    @patch("pyrun_jupyter.kernel.requests.request")
    def test_token_sent_as_header_and_query_parameter(self, mock_request):
        """Proxies such as Kaggle's ignore the header and need ?token=."""
        KernelManager("http://localhost:8888", token="xxx").list_kernels()

        kwargs = mock_request.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "token xxx"
        assert kwargs["params"] == {"token": "xxx"}

    @patch("pyrun_jupyter.kernel.requests.request")
    def test_no_token_no_query_parameter(self, mock_request):
        """Without a token nothing is added to the URL."""
        KernelManager("http://localhost:8888").list_kernels()

        assert mock_request.call_args.kwargs["params"] == {}

    @patch("pyrun_jupyter.kernel.requests.request")
    def test_token_redacted_from_http_error(self, mock_request):
        """The token is part of the request URL, so it must not leak into errors."""
        response = Mock(status_code=404)
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error: Not Found for url: http://localhost:8888/api/kernels?token=xxx"
        )
        mock_request.return_value = response

        with pytest.raises(ConnectionError) as exc_info:
            KernelManager("http://localhost:8888", token="xxx").list_kernels()

        assert "xxx" not in str(exc_info.value)
        assert exc_info.value.__suppress_context__
