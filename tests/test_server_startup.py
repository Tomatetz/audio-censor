import unittest
import errno
from unittest.mock import Mock, patch

import web_gui


class ServerStartupTests(unittest.TestCase):
    def test_reuses_existing_stream_censor_server(self):
        occupied = OSError(errno.EADDRINUSE, "Address already in use")
        with patch.object(
            web_gui.ReusableThreadingHTTPServer,
            "__new__",
            side_effect=occupied,
        ):
            with patch.object(web_gui, "is_stream_censor_server", return_value=True):
                server, port, existing = web_gui.find_or_create_server(
                    start_port=8765,
                    attempts=1,
                )
        self.assertIsNone(server)
        self.assertEqual(port, 8765)
        self.assertTrue(existing)

    def test_uses_next_port_when_foreign_app_owns_first(self):
        occupied = OSError(errno.EADDRINUSE, "Address already in use")
        server = Mock()
        with patch(
            "web_gui.ReusableThreadingHTTPServer",
            side_effect=[occupied, server],
        ):
            with patch.object(web_gui, "is_stream_censor_server", return_value=False):
                result, port, existing = web_gui.find_or_create_server(
                    start_port=8765,
                    attempts=2,
                )
        self.assertIs(result, server)
        self.assertEqual(port, 8766)
        self.assertFalse(existing)


if __name__ == "__main__":
    unittest.main()
