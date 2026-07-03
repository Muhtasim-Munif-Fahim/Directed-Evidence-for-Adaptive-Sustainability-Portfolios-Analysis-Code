from __future__ import annotations

import json
import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_SOURCE = {
    "audit_data.py",
    "build_evidence_registry.py",
    "download_world_bank.py",
    "prepare_external_validation.py",
    "result_contract.py",
    "run_panel_granger_network.py",
    "run_panel_local_projections.py",
    "run_external_validation.py",
    "run_external_validation_full.py",
    "run_selection_table.py",
}

FORBIDDEN_SOURCE_NAMES = {
    "check_" + "manu" + "script_consistency.py",
    "generate_" + "network_tables.py",
    "run_" + "supp" + "lement_audits.py",
}

FORBIDDEN_FRAGMENTS = [
    "J:" + "\\",
    "C:" + "\\Users\\Admin",
    "Survey" + " Research",
    "gho" + "_",
    "available from the corresponding author " + "upon reasonable request",
    "check_" + "manu" + "script",
    "generate_" + "network_tables",
    "run_" + "supp" + "lement",
    "outputs/" + "fig" + "ures",
    "pa" + "per/" + "generated",
]


class PublicReleaseChecks(unittest.TestCase):
    def test_expected_core_files_exist(self) -> None:
        present = {path.name for path in (ROOT / "src").glob("*.py")}
        self.assertTrue(EXPECTED_SOURCE.issubset(present))
        self.assertTrue((ROOT / "README.md").exists())
        self.assertTrue((ROOT / "data" / "SOURCES.md").exists())
        self.assertTrue((ROOT / "config" / "study_design.json").exists())

    def test_no_rendering_or_submission_helpers_are_tracked(self) -> None:
        present = {path.name for path in (ROOT / "src").glob("*.py")}
        self.assertFalse(any(name.startswith("fig" + "_") for name in present))
        self.assertTrue(FORBIDDEN_SOURCE_NAMES.isdisjoint(present))

    def test_protocol_matches_current_public_analysis(self) -> None:
        protocol = json.loads(
            (ROOT / "config" / "study_design.json").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["protocol_version"], "2026-06-07.1")
        self.assertEqual(len(protocol["sample"]["goals"]), 17)
        self.assertFalse(protocol["claim_rules"]["causal_language_allowed"])

    def test_python_sources_compile(self) -> None:
        for path in sorted((ROOT / "src").glob("*.py")):
            with self.subTest(path=path.name):
                py_compile.compile(str(path), doraise=True)

    def test_no_local_paths_or_private_tokens_in_tracked_text(self) -> None:
        checked_suffixes = {".py", ".md", ".json", ".txt", ".gitignore"}
        for path in ROOT.rglob("*"):
            if ".git" in path.parts or not path.is_file() or path.suffix not in checked_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for fragment in FORBIDDEN_FRAGMENTS:
                with self.subTest(path=path.relative_to(ROOT), fragment=fragment):
                    self.assertNotIn(fragment, text)

    def test_large_data_outputs_are_not_tracked(self) -> None:
        for folder in ("data/raw", "data/processed", "outputs"):
            files = [
                path
                for path in (ROOT / folder).rglob("*")
                if path.is_file() and path.name != ".gitkeep"
            ]
            self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
