"""``spinlabs plot t1 --query`` resolves runs from the registry.

It shares the ``--defect/--cell-size/--method/--order/--pert-scale`` filter
flags with ``run --query``. Rows are written through :func:`register_run` and
read back through :func:`find_runs` so the tests exercise the exact format the
pipeline produces (``order`` as a JSON list such as ``[1, 2]``).
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
    run_id = registry.register_run(
        conn,
        run_dir=run_dir,
        method="all_bands",
        init_state="ms_1",
        defect="NV",
        cell=512,
        order=[1, 2],
        pert=0.025,
        npz_files=[run_dir / "t1_relaxation.npz"],
    )
    registry.finish_run(conn, run_id, "done")
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
            "--order", "1", "2", "--pert-scale", "0.025", "-o", str(tmp_path / "out.png"),
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
    dirs = registry.find_runs(conn, [("order", [1, 2])])
    assert dirs == [run_dir]


def _add_run(conn, run_dir, *, order=1, status="done", started_at=None, **kw):
    run_id = registry.register_run(
        conn, run_dir=run_dir, defect="NV", cell=512, order=order, pert=0.025,
        method="all_bands", init_state="ms_1", **kw,
    )
    if status != "running":
        registry.finish_run(conn, run_id, status)
    if started_at is not None:
        conn.execute("UPDATE runs SET started_at = ? WHERE id = ?", (started_at, run_id))
        conn.commit()
    return run_id


def test_order_12_roundtrip(tmp_path, monkeypatch):
    """--order 12 (combined-order convention) must match the row register_run wrote."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_a", order=1)
    _add_run(conn, tmp_path / "r_c", order=12)
    dirs = registry.find_runs(conn, [("order", "12")])
    assert [d.name for d in dirs] == ["r_c"]
    assert registry.find_runs(conn, [("order", "1")])[0].name == "r_a"


def test_order_list_roundtrip(tmp_path, monkeypatch):
    """A pipeline-registered order list ([1, 2]) must be queryable as '1 2'."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_c", order=[1, 2])
    assert registry.find_runs(conn, [("order", "1 2")])[0].name == "r_c"
    assert registry.find_runs(conn, [("order", "12")])[0].name == "r_c"


def test_latest_returns_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_oct01", started_at="2026-10-01 12:00:00")
    _add_run(conn, tmp_path / "r_oct05", started_at="2026-10-05 12:00:00")
    _add_run(conn, tmp_path / "r_oct10", started_at="2026-10-10 12:00:00")
    dirs = registry.find_runs(conn, [("defect", "NV")], latest=True)
    assert [d.name for d in dirs] == ["r_oct10"]
    # Non-latest queries keep returning all matching runs (oldest first).
    assert [d.name for d in registry.find_runs(conn, [("defect", "NV")])] == [
        "r_oct01", "r_oct05", "r_oct10",
    ]


def test_latest_ignores_running_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_done", started_at="2026-10-01 12:00:00")
    _add_run(conn, tmp_path / "r_running", status="running", started_at="2026-10-11 12:00:00")
    dirs = registry.find_runs(conn, [("defect", "NV")], latest=True)
    assert [d.name for d in dirs] == ["r_done"]


def test_no_match_invalid_order_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_a", order=1)
    with pytest.raises(SystemExit):
        registry.find_runs(conn, [("order", "3")])


@pytest.mark.parametrize("subcmd", ["coupling", "rates"])
def test_plot_run_subcommand_query(subcmd, registry_db, tmp_path, monkeypatch, capsys):
    """plot coupling/rates --query resolve registry runs like plot t1."""
    monkeypatch.chdir(tmp_path)
    from spinlabs.cli import build_parser

    called = []

    def fake_plot(run_dir, out_dir, fmt, dpi, show):
        out_dir.mkdir(parents=True, exist_ok=True)
        called.append(run_dir)

    import spinlabs.cli.plotting as plotting
    monkeypatch.setattr(plotting, "plot_run_coupling", fake_plot)
    monkeypatch.setattr(plotting, "plot_run_rates", fake_plot)
    args = build_parser().parse_args(["plot", subcmd, "--query", "--defect", "NV", "--cell-size", "512"])
    args.func(args)
    assert len(called) == 1
    assert called[0].name == "NV_512_pert_0.025"
    assert (tmp_path / "figures" / subcmd / "NV_512_pert_0.025").exists()


@pytest.mark.parametrize("subcmd", ["coupling", "rates", "t1"])
def test_plot_query_no_match_errors(subcmd, registry_db, monkeypatch):
    from spinlabs.cli import build_parser

    args = build_parser().parse_args(["plot", subcmd, "--query", "--defect", "ClV"])
    with pytest.raises(SystemExit, match="no completed runs"):
        args.func(args)


@pytest.mark.parametrize("subcmd", ["coupling", "rates"])
def test_plot_path_and_query_conflict(subcmd, registry_db):
    from spinlabs.cli import build_parser

    args = build_parser().parse_args(["plot", subcmd, "some_dir", "--query", "--defect", "NV"])
    with pytest.raises(SystemExit, match="not both"):
        args.func(args)
