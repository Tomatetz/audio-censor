import unittest
from unittest.mock import Mock, patch

import web_gui


class WebShutdownTests(unittest.TestCase):
    def test_shutdown_stops_filter_and_server(self):
        server = Mock()
        with patch.object(web_gui, "SERVER", server):
            with patch.object(web_gui.STATE, "stop") as stop:
                with patch.object(web_gui.STATE, "process", None):
                    web_gui.shutdown_application()
        stop.assert_called_once()
        server.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
