"""Non-training checks for the bounded analysis pilot and Windows launcher."""
import importlib.util
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'scripts/adaptation_analysis/pilot.py'
LAUNCHER = ROOT / 'run_adaptation_pilot.ps1'
POWERSHELL = shutil.which('powershell') or shutil.which('pwsh')


class PilotArgumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PILOT.exists():
            raise unittest.SkipTest('Pilot implementation is not present yet')
        spec = importlib.util.spec_from_file_location('adaptation_pilot_cli_under_test', PILOT)
        cls.pilot = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.pilot
        spec.loader.exec_module(cls.pilot)

    def test_three_pc_assignments_parse(self):
        for pc in ('pc1', 'pc2', 'pc3'):
            with self.subTest(pc=pc):
                args = self.pilot.build_parser().parse_args(['--pc', pc])
                self.assertEqual(args.pc, pc)
                self.assertEqual(args.samples, 256)
                self.assertEqual(list(args.seeds), [42, 123, 456])

    def test_explicit_bounded_job_and_non_mutating_flags(self):
        args = self.pilot.build_parser().parse_args([
            '--pc', 'pc3', '--samples', '8', '--seeds', '42',
            '--device', 'cpu', '--output-root', 'results/experiments/pilot_cli_check',
            '--check-only', '--self-check',
        ])
        self.assertEqual(args.samples, 8)
        self.assertEqual(list(args.seeds), [42])
        self.assertEqual(args.device, 'cpu')
        self.assertTrue(args.check_only)
        self.assertTrue(args.self_check)

    def test_unknown_pc_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.pilot.build_parser().parse_args(['--pc', 'pc4'])

    def test_preflight_rejects_invalid_sample_budget_without_loading_artifacts(self):
        study = {'max_samples': 512, 'seeds': [42, 123, 456]}
        for samples in ('0', '-1', '513'):
            with self.subTest(samples=samples):
                args = self.pilot.build_parser().parse_args([
                    '--pc', 'pc1', '--samples', samples, '--device', 'cpu',
                ])
                with self.assertRaisesRegex(ValueError, '--samples'):
                    self.pilot.preflight(args, study)

    def test_preflight_rejects_duplicate_or_unregistered_seeds_without_loading_artifacts(self):
        study = {'max_samples': 512, 'seeds': [42, 123, 456]}
        for seeds in (['42', '42'], ['99']):
            with self.subTest(seeds=seeds):
                args = self.pilot.build_parser().parse_args([
                    '--pc', 'pc1', '--device', 'cpu', '--seeds', *seeds,
                ])
                with self.assertRaisesRegex(ValueError, '--seeds'):
                    self.pilot.preflight(args, study)

    def test_help_works_from_another_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='adaptation-cli-help-') as folder:
            result = subprocess.run(
                [sys.executable, str(PILOT), '--help'], cwd=folder,
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--check-only', result.stdout)
        self.assertIn('--self-check', result.stdout)
        self.assertIn('--pc', result.stdout)


class LauncherChecks(unittest.TestCase):
    def test_launcher_uses_repository_root_and_project_environment(self):
        source = LAUNCHER.read_text(encoding='utf-8')
        self.assertIn('$PSScriptRoot', source)
        self.assertIn("'.venv\\Scripts\\python.exe'", source)
        self.assertIn("'.ven\\Scripts\\python.exe'", source)
        self.assertIn('Push-Location -LiteralPath $projectRoot', source)
        self.assertIn('Pop-Location', source)
        self.assertNotIn('Start-Process', source)

    @unittest.skipUnless(POWERSHELL, 'PowerShell is unavailable')
    def test_powershell_syntax(self):
        quoted = str(LAUNCHER).replace("'", "''")
        command = (
            '$tokens=$null; $errors=$null; '
            f"[System.Management.Automation.Language.Parser]::ParseFile('{quoted}', "
            '[ref]$tokens, [ref]$errors) | Out-Null; '
            'if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }'
        )
        result = subprocess.run(
            [POWERSHELL, '-NoProfile', '-Command', command],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(POWERSHELL, 'PowerShell is unavailable')
    def test_self_check_launcher_from_another_directory_writes_no_results(self):
        if not PILOT.exists():
            self.skipTest('Pilot implementation is not present yet')
        with tempfile.TemporaryDirectory(prefix='adaptation-launcher-check-') as folder:
            output = Path(folder) / 'must-not-be-created'
            result = subprocess.run([
                POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', str(LAUNCHER), '-PC', 'pc2', '-SelfCheck',
                '-Samples', '3', '-Device', 'cpu', '-OutputRoot', str(output),
            ], cwd=folder, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(output.exists(), 'Synthetic self-check wrote a results directory')


if __name__ == '__main__':
    unittest.main()
