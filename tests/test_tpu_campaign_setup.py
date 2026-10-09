"""Exercise the real setup gate with fake gcloud calls; never rent hardware."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CampaignSetupTests(unittest.TestCase):
    def test_queue_rejects_a_missing_slice_instead_of_guessing_one_worker(self):
        source = (ROOT / "scripts/tpu_collection/tpu_queue.sh").read_text()
        function = "derive_workers() {" + source.split("derive_workers() {", 1)[1].split("\nWORKERS_DERIVED=", 1)[0]
        script = r'''
set -uo pipefail
PROJECT=test ZONE=test
slice_nodes(){ echo slice0; echo slice1; }
gcloud(){
  case "$*" in
    *slice0*) echo '10.0.0.1;10.0.0.2' ;;
    *slice1*) return 1 ;;
  esac
}
''' + function + '\nderive_workers\n'
        result = subprocess.run(["bash", "-c", script], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")

    def run_setup(self, workers, successful, ssh_rc=0, describe_rc=0):
        source = (ROOT / "scripts/tpu_collection/orchestration/campaign.sh").read_text()
        function = "run_scope(){" + source.split("run_scope(){", 1)[1].split("\nEND=", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, TEST_DIR=tmp, TEST_WORKERS=str(workers),
                       TEST_SUCCESSFUL=str(successful), TEST_SSH_RC=str(ssh_rc),
                       TEST_DESCRIBE_RC=str(describe_rc))
            # Keep the function's scratch logs inside this test's directory.
            function = function.replace("/tmp/setup.", '"$TEST_DIR"/setup.')
            function = function.replace("/tmp/q.", '"$TEST_DIR"/q.')
            script = r'''
set -uo pipefail
PROJECT=test SESSION=test BUCKET=gs://test GCS=gs://test
REPS=1 DUR=1 PER_RUN_CAP_SEC=1
say(){ :; }
mark_collected(){ :; }
gcloud(){
  case "$*" in
    *" tpu-vm describe "*)
      [ "$TEST_DESCRIBE_RC" = 0 ] || return "$TEST_DESCRIBE_RC"
      for ((j=0;j<TEST_WORKERS;j++)); do echo "10.0.0.$((j+1))"; done ;;
    *" tpu-vm ssh "*)
      for ((j=0;j<TEST_SUCCESSFUL;j++)); do
        echo "JAXOK 0.4.35 8"; echo "SETUP_OK_worker$j"
      done
      return "$TEST_SSH_RC" ;;
    *) return 0 ;;
  esac
}
timeout(){ touch "$TEST_DIR/launched"; echo 'done: 1 failed: 0'; }
''' + function + '\nrun_scope test-zone test-slice 1 tpuT16 gate\n'
            result = subprocess.run(["bash", "-c", script], env=env,
                                    capture_output=True, text=True, timeout=10)
            return result.returncode, (Path(tmp) / "launched").exists()

    def test_zero_workers_blocks_launch(self):
        self.assertEqual(self.run_setup(2, 0), (1, False))

    def test_partial_multihost_slice_blocks_launch(self):
        self.assertEqual(self.run_setup(2, 1), (1, False))

    def test_ssh_failure_blocks_launch_even_with_success_messages(self):
        self.assertEqual(self.run_setup(2, 2, ssh_rc=1), (1, False))

    def test_unknown_worker_count_blocks_launch(self):
        self.assertEqual(self.run_setup(2, 2, describe_rc=1), (1, False))

    def test_all_workers_allows_launch(self):
        self.assertEqual(self.run_setup(2, 2), (0, True))


if __name__ == "__main__":
    unittest.main()
