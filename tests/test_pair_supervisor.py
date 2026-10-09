"""Exercise supervision decisions without cloud access or provisioning."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SupervisorTests(unittest.TestCase):
    def run_supervisor(self, done=False, running=0, listing_rc=0):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, TEST_DIR=tmp, TEST_DONE=str(int(done)),
                       TEST_RUNNING=str(running), TEST_LISTING_RC=str(listing_rc),
                       MAX_CYCLES="1", RUN_STAMP="testa", CHECK_SECS="0")
            prefix = r'''
sleep(){ :; }
gcloud(){
  case "$*" in
    "storage ls "*) [ "$TEST_DONE" = 1 ] ;;
    "compute instances list "*)
      [ "$TEST_LISTING_RC" = 0 ] || return "$TEST_LISTING_RC"
      for ((j=0;j<TEST_RUNNING;j++)); do echo "spar-spot-$j"; done ;;
    *) echo "unexpected cloud operation" >&2; return 90 ;;
  esac
}
bash(){ touch "$TEST_DIR/provisioned"; return 0; }
'''
            source = (ROOT / "scripts/gcp/pair_supervisor.sh").read_text()
            source = source.replace('"/tmp/supervisor_${RUN_STAMP}.log"',
                                    '"$TEST_DIR/supervisor.log"')
            result = subprocess.run(["bash", "-c", prefix + source], env=env,
                                    cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            return (Path(tmp) / "provisioned").exists()

    def test_completed_round_does_not_rent_another_pair(self):
        self.assertFalse(self.run_supervisor(done=True))

    def test_failed_inventory_query_does_not_rent_a_pair(self):
        self.assertFalse(self.run_supervisor(listing_rc=1))

    def test_healthy_pair_is_left_running(self):
        self.assertFalse(self.run_supervisor(running=2))

    def test_preempted_incomplete_round_relaunches(self):
        self.assertTrue(self.run_supervisor())


if __name__ == "__main__":
    unittest.main()
