import unittest
import time
from api.security import sign_jwt, verify_jwt, is_rate_limited, validate_repo_url


class TestSecurity(unittest.TestCase):
    def test_jwt_flow(self):
        payload = {"user": "alice"}
        token = sign_jwt(payload)
        
        verified = verify_jwt(token)
        self.assertIsNotNone(verified)
        self.assertEqual(verified["user"], "alice")

    def test_invalid_jwt(self):
        self.assertIsNone(verify_jwt("invalid.token.structure"))
        self.assertIsNone(verify_jwt("a.b.c"))

    def test_validate_repo_url(self):
        # Valid cases
        self.assertTrue(validate_repo_url("https://github.com/user/repo"))
        self.assertTrue(validate_repo_url("git@github.com:user/repo.git"))
        self.assertTrue(validate_repo_url("local_path/to/repo"))

        # Invalid cases
        self.assertFalse(validate_repo_url("https://github.com/user/repo; rm -rf /"))
        self.assertFalse(validate_repo_url("git@github.com:user/repo.git && whoami"))


if __name__ == "__main__":
    unittest.main()
