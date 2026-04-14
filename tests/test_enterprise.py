import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil

from config import config
from api.enterprise import log_audit_action, get_audit_logs, verify_role_access, DB_PATH
from fastapi.testclient import TestClient
from fastapi import HTTPException
from api.server import app


class TestEnterprise(unittest.TestCase):
    def setUp(self):
        self.old_security = config.security_enabled
        config.security_enabled = True
        
        # Isolate database
        self.test_dir = tempfile.mkdtemp()
        self.old_db_path = DB_PATH
        import api.enterprise
        api.enterprise.DB_PATH = os.path.join(self.test_dir, "enterprise_test.db")

    def tearDown(self):
        config.security_enabled = self.old_security
        import api.enterprise
        api.enterprise.DB_PATH = self.old_db_path
        shutil.rmtree(self.test_dir)

    def test_audit_logs(self):
        # Log actions
        log_audit_action("admin-user", "test_action", "details_here")
        log_audit_action("dev-user", "dev_action", "dev_details")
        
        logs = get_audit_logs()
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["user"], "dev-user")
        self.assertEqual(logs[1]["user"], "admin-user")

    def test_verify_role_access(self):
        # Admin Key
        role = verify_role_access("admin-key", ["admin"])
        self.assertEqual(role, "admin")

        # Developer Key
        role2 = verify_role_access("Bearer dev-key", ["admin", "developer"])
        self.assertEqual(role2, "developer")

        # Missing Key
        with self.assertRaises(HTTPException) as cm:
            verify_role_access(None, ["admin"])
        self.assertEqual(cm.exception.status_code, 401)

        # Invalid Key
        with self.assertRaises(HTTPException) as cm:
            verify_role_access("invalid-key", ["admin"])
        self.assertEqual(cm.exception.status_code, 403)

        # Insufficient Permissions
        with self.assertRaises(HTTPException) as cm:
            verify_role_access("viewer-key", ["admin", "developer"])
        self.assertEqual(cm.exception.status_code, 403)

    def test_enterprise_endpoints(self):
        client = TestClient(app, raise_server_exceptions=False)
        
        # Test audit logs endpoint (admin allowed)
        resp_admin = client.get("/enterprise/audit-logs", headers={"Authorization": "admin-key"})
        self.assertEqual(resp_admin.status_code, 200)

        # Test audit logs endpoint (developer forbidden)
        resp_dev = client.get("/enterprise/audit-logs", headers={"Authorization": "dev-key"})
        self.assertEqual(resp_dev.status_code, 403)

        # Test collections endpoint (developer allowed)
        resp_coll = client.get("/enterprise/collections", headers={"Authorization": "dev-key"})
        self.assertEqual(resp_coll.status_code, 200)
        self.assertIn("default-collection", resp_coll.json())


if __name__ == "__main__":
    unittest.main()
