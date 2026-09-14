import bootstrap  # noqa
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from namd_analysis.cli import main
from namd_analysis.comparison import compare_schemes, compare_runs
from namd_analysis.fitting import bootstrap_single_exponential, FitError
from namd_analysis.kinetics import build_rate_matrix, propagate
from namd_analysis.populations import StateMap, load_population_set, group_series, InputMismatchError, ConfigError
from namd_analysis.provenance import launcher_manifests
from synthetic import write_shprop_set, kinetic_config


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mapping = kinetic_config(["VBM", "CBM"], "VBM")
        self.map_path = self.root / "map.json"
        self.map_path.write_text(json.dumps(self.mapping))

    def tearDown(self):
        self.tmp.cleanup()

    def test_bootstrap_recovers_known_spread_reproducibly(self):
        t = np.linspace(0, 5, 80)
        files = np.stack([.9*np.exp(-t/tau) for tau in [1.7,1.9,2.1,2.3]])
        a = bootstrap_single_exponential(t, files, n_resamples=40, seed=2)
        b = bootstrap_single_exponential(t, files, n_resamples=40, seed=2)
        self.assertEqual(a, b)
        self.assertEqual(a["successful"], 40)
        self.assertLess(a["tau_interval"][0], 2)
        self.assertGreater(a["tau_interval"][1], 2)

    def test_bootstrap_missing_spread_has_no_confidence_interval(self):
        t = np.linspace(0, 5, 30)
        for count in [1,3]:
            result = bootstrap_single_exponential(t, np.tile(np.exp(-t), (count,1)), n_resamples=20)
            self.assertIsNone(result["tau_interval"])
            self.assertTrue(result["status"].startswith("unavailable"))

    def test_failed_bootstrap_samples_reported(self):
        t = np.linspace(0, 5, 30)
        files = np.stack([np.exp(-t), 1-np.exp(-t)])
        result = bootstrap_single_exponential(t, files, n_resamples=40)
        self.assertGreater(result["failed"], 0)
        self.assertIsNone(result["tau_interval"])

    def test_bootstrap_rejects_too_few_resamples(self):
        with self.assertRaises(FitError):
            bootstrap_single_exponential(np.arange(4), np.ones((2,4)), n_resamples=2)

    def test_scheme_ranking_uses_independent_population_contrasts(self):
        t = np.linspace(0, 3, 60)
        groups = ["A", "B", "C"]
        K = build_rate_matrix([2, .5], [(0,1),(1,2)], 3)
        obs = propagate(K, [1,0,0], t)
        report, fits = compare_schemes(t, obs, groups,
                                      {"sequential": "A->B,B->C", "wrong": "A->B,A->C"}, n_starts=2)
        self.assertEqual(report["lowest_score_candidate"], "sequential")
        self.assertEqual(report["candidates"][0]["n_observations"], 59*2)
        self.assertEqual(report["candidates"][0]["k_parameters"], 3)
        self.assertEqual(len(fits), 2)

    def test_identical_graphs_and_bad_observations_rejected(self):
        t = np.linspace(0,1,20)
        obs = np.column_stack([np.exp(-t),1-np.exp(-t)])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            compare_schemes(t,obs,["A","B"],{"a":"A->B","b":"A->B"})
        with self.assertRaisesRegex(ValueError,"conserved"):
            compare_schemes(t,obs*2,["A","B"],{"a":"A->B","b":"B->A"})

    def make_manifest(self):
        write_shprop_set(self.root / "fast", n_files=2, nsteps=40, tau_fs=10000)
        write_shprop_set(self.root / "slow", n_files=2, nsteps=40, tau_fs=30000)
        m = self.root / "runs.json"
        m.write_text(json.dumps({"reference":"fast", "runs":[
            {"label":"fast","initial_state":"perovskite","config":"map.json","files":["fast/SHPROP.*"]},
            {"label":"slow","initial_state":"BCF","config":"map.json","files":["slow/SHPROP.*"]}]}))
        return m

    def test_run_comparison_preserves_initial_conditions(self):
        m = self.make_manifest()
        result, t, curves = compare_runs(m,start_ns=.005,end_ns=.020)
        self.assertEqual(len(t),16)
        self.assertEqual(result["runs"][1]["initial_state_description"],"BCF")
        diff = next(row for row in result["differences"] if row["group"]=="CBM")
        self.assertGreater(diff["final_difference"],0)
        self.assertEqual(len(result["inputs"]),7)

    def test_comparison_rejects_no_common_grid(self):
        m = self.make_manifest()
        for p in (self.root/"slow").glob("SHPROP.*"):
            data=np.loadtxt(p); data[:,0] += .1; np.savetxt(p,data)
        with self.assertRaisesRegex(ValueError,"exact shared"):
            compare_runs(m)

    def test_group_sem_preserves_anticorrelation(self):
        paths=write_shprop_set(self.root/"sem", n_files=2, nsteps=20)
        data=np.loadtxt(paths[1]); data[:,2:]=data[:,2:][:,::-1]; np.savetxt(paths[1],data)
        mapping=StateMap.from_dict(dict(self.mapping,groups={"all":[2,3]},recombined_group=None))
        series=group_series(load_population_set(paths,mapping),mapping)
        np.testing.assert_allclose(series[0].sem,0,atol=1e-15)

    def test_complete_map_and_per_file_conservation_enforced(self):
        with self.assertRaises(ConfigError):
            StateMap.from_dict(dict(self.mapping,groups={"VBM":[2]}))
        for bad in [-1, 2.5, True]:
            with self.assertRaises(ConfigError):
                StateMap.from_dict(dict(self.mapping,time_column=bad))
        paths=write_shprop_set(self.root/"bad",n_files=2,nsteps=20)
        for sign,p in zip([1,-1],paths):
            data=np.loadtxt(p); data[:,2:]=.5+sign*.01; np.savetxt(p,data)
        with self.assertRaisesRegex(InputMismatchError,"every file"):
            load_population_set(paths,StateMap.from_dict(self.mapping))

    def test_launcher_provenance_imports_but_does_not_trust_paths(self):
        raw=self.root/"SHPROP.1";raw.write_text("0 0 1\n")
        manifest=self.root/"hefei_manifest.json"
        manifest.write_text(json.dumps({"artifact":"/does/not/exist", "engine":"dish"}))
        invalid=self.root/"nac_manifest.json"; invalid.write_text("not json")
        records=launcher_manifests([raw])
        self.assertEqual(len(records),2)
        self.assertEqual(records[0]["status"],"imported")
        self.assertEqual(records[1]["status"],"invalid")
        self.assertIn("sha256", records[0]["input"])

    def test_cli_compare_runs_and_decay_bootstrap(self):
        m=self.make_manifest()
        self.assertEqual(main(["compare-runs","--manifest",str(m),"--out",str(self.root/"comparison")]),0)
        self.assertTrue((self.root/"comparison/comparison.png").exists())
        self.assertEqual(main(["populations","--files",str(self.root/"fast/SHPROP.*"),"--config",str(self.map_path),"--fit-group","CBM","--bootstrap","20","--out",str(self.root/"fit")]),0)
        report=json.loads((self.root/"fit/report.json").read_text())
        self.assertEqual(report["fit_uncertainty"]["status"],"unavailable_identical_files")
        self.assertIn("config_input",report)

    def test_cli_scheme_comparison_records_config_and_rank(self):
        paths=write_shprop_set(self.root/"kinetics",n_files=2,nsteps=30)
        schemes=self.root/"schemes.json";schemes.write_text(json.dumps({"decay":"CBM->VBM","backward":"VBM->CBM"}))
        self.assertEqual(main(["compare-schemes","--files",str(paths[0]),"--config",str(self.map_path),"--schemes",str(schemes),"--out",str(self.root/"ranking")]),0)
        report=json.loads((self.root/"ranking/report.json").read_text())
        self.assertEqual(report["lowest_score_candidate"],"decay")
        self.assertTrue((self.root/"ranking/schemes.csv").exists())
