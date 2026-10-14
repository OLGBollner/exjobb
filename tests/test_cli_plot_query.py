"""``spinlabs plot t1 --query`` resolves runs from the registry.

It shares the ``--defect/--cell-size/--method/--order/--pert-scale`` filter
flags with ``run --query``; ``order`` is an SQL reserved word and the ``order``
column is stored as a JSON list, so the resolution must quote columns and
match list values.
"""


import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.cli.plotting as cli_plot
import spinlabs.registry as registry


def _parse(argv):
    parser = cli.build_parser()
    return parser.parse_args(["plot", "t1", *argv])


@pytest.fixture
def registry_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    run_dir = tmp_path / "runs" / "NV_512_pert_0.025"
    run_dir.mkdir(parents=True)
    T = np.linspace(100.0, 500.0, 5)
    np.savez(
        run_dir / "t1_relaxation.npz",
        temperatures=T,
        t1_fit=T,
        t1_eigenval=T,
        defect="NV",
        cell_size="512",
        calc_method="all_bands",
        init_state="ms_1",
        times=T,
        P=T,
        rates=T,
        coupling=T,
        modes=np.arange(3),
        freqs=T[:3],
        couplings=T[:3],
    )
    conn.execute(
        'insert into runs (run_dir, status, defect, cell, "order", pert, method, n_dims,'
        " started_at, finished_at) values (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
        (str(run_dir), "done", "NV", 512, "[1]", 0.025, "all_bands", "1+2"),
    )
    conn.commit()
    return conn, run_dir


def _run_cli(monkeypatch, argv, tmp_path):
    """Execute the t1 subcommand with plotting stubbed; return the resolved inputs."""
    saved = {}
    monkeypatch.setattr(cli_plot, "plot_t1", lambda paths, **k: saved.update(paths=paths) or (None, None))
    cli_plot._run_t1(_parse(argv))
    return saved["paths"]


def test_query_resolves_single_run(registry_db, monkeypatch, tmp_path):
    conn, run_dir = registry_db
    paths = _run_cli(
        monkeypatch,
        [
            "--query", "--defect", "NV", "--cell-size", "512", "--method", "all",
            "--order", "1", "--pert-scale", "0.025", "-o", str(tmp_path / "out.png"),
        ],
        tmp_path,
    )
    assert paths == [run_dir / "t1_relaxation.npz"]


def test_query_no_match_is_error(registry_db, monkeypatch, tmp_path):
    conn, _ = registry_db
    with pytest.raises(SystemExit, match="no completed runs"):
        _run_cli(
            monkeypatch,
            ["--query", "--defect", "ClV", "-o", str(tmp_path / "out.png")],
            tmp_path,
        )


def test_query_and_inputs_are_mutually_exclusive(monkeypatch):
    with pytest.raises(SystemExit, match="not both"):
        _run_cli(monkeypatch, ["--query", "some.npz"], None)



def test_order_column_is_quoted(registry_db):
    conn, run_dir = registry_db
    dirs = registry.find_runs(conn, [("order", 1)])
    assert dirs == [run_dir]
