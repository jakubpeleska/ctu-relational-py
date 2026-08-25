import shutil
from pathlib import Path

import pytest

pydot = pytest.importorskip("pydot")
pytest.importorskip("jinja2")

from redelex.db import RemoteDBInterface  # noqa: E402
from redelex.db.visualize import visualize_db  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("dot") is None,
    reason="rendering the schema to SVG needs the Graphviz 'dot' executable",
)


def test_visualize_db_writes_svg(sqlite_db_url, tmp_path):
    """Regression: the .dot file was read before the writes were flushed, so
    pydot parsed an empty (or truncated) file."""
    with RemoteDBInterface(sqlite_db_url) as iface:
        out = visualize_db(iface, "testdb", str(tmp_path))

    assert out == str(tmp_path / "testdb.svg")
    svg = (tmp_path / "testdb.svg").read_text()
    assert svg.lstrip().startswith(("<?xml", "<svg"))
    # Every table of the fixture must show up as a node.
    for table in ["customer", "product", "orders", "events"]:
        assert table in svg


def test_visualize_db_hide_columns(sqlite_db_url, tmp_path):
    with RemoteDBInterface(sqlite_db_url) as iface:
        out = visualize_db(iface, "nocols", str(tmp_path), hide_columns=True)

    svg = Path(out).read_text()
    assert "customer" in svg
    # Column names are omitted when the fields are disabled.
    assert "signup_date" not in svg
