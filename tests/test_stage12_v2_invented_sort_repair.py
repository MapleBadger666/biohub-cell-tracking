from __future__ import annotations

import ast
import hashlib

from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

PORTABLE = (
    ROOT
    / "src/biohub_cell_tracking/stage12_portable_current_best.py"
)

HISTORICAL = (
    ROOT
    / "scripts/run_stage12_fold0_v2_action_lock.py"
)

TARGET_BUILDER = (
    "build_portable_mutual_cont_v2_actions"
)

EXPECTED_PORTABLE_SHA = (
    "a1fe0c9e5cd5e20a2ae5921bd4b4c55d078022c3348d5945bffa501ac3ff36ed"
)

EXPECTED_HISTORICAL_SHA = (
    "8e2684669baeaf8bc0d4b889f5c6e213a9127cf09e36c3d76ea2b3d77c2f4ef5"
)

EXPECTED_COLUMNS = (
    ['dataset', 't', 'source_pred_node_id', 'target_pred_node_id']
)

EXPECTED_KIND = (
    "mergesort"
)


def _sha(path: Path) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _functions(tree: ast.Module):

    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    }


def _literal_sequence(node):

    if not isinstance(
        node,
        (
            ast.List,
            ast.Tuple,
        ),
    ):
        return None

    values = []

    for item in node.elts:

        if not (
            isinstance(
                item,
                ast.Constant,
            )
            and isinstance(
                item.value,
                str,
            )
        ):
            return None

        values.append(
            item.value
        )

    return values


def _columns(call):

    if call.args:

        result = _literal_sequence(
            call.args[0]
        )

        if result is not None:
            return result

    for kw in call.keywords:

        if kw.arg == "by":

            result = _literal_sequence(
                kw.value
            )

            if result is not None:
                return result

    return None


def _kind(call):

    for kw in call.keywords:

        if (
            kw.arg == "kind"
            and isinstance(
                kw.value,
                ast.Constant,
            )
        ):
            return kw.value.value

    return None


def _exact_sorts(tree):

    result = []

    for call in ast.walk(
        tree
    ):

        if not (
            isinstance(
                call,
                ast.Call,
            )
            and isinstance(
                call.func,
                ast.Attribute,
            )
            and call.func.attr
            == "sort_values"
        ):
            continue

        if (
            _columns(call)
            == EXPECTED_COLUMNS
            and _kind(call)
            == EXPECTED_KIND
        ):
            result.append(
                call
            )

    return result


def test_portable_post_repair_sha():

    assert (
        _sha(PORTABLE)
        == EXPECTED_PORTABLE_SHA
    )


def test_historical_v2_sha():

    assert (
        _sha(HISTORICAL)
        == EXPECTED_HISTORICAL_SHA
    )


def test_v2_action_builder_still_exists():

    tree = ast.parse(
        PORTABLE.read_text(
            encoding="utf-8"
        )
    )

    assert (
        TARGET_BUILDER
        in _functions(tree)
    )


def test_invented_sort_absent_portable():

    tree = ast.parse(
        PORTABLE.read_text(
            encoding="utf-8"
        )
    )

    assert (
        _exact_sorts(tree)
        == []
    )


def test_equivalent_sort_absent_historical():

    tree = ast.parse(
        HISTORICAL.read_text(
            encoding="utf-8"
        )
    )

    assert (
        _exact_sorts(tree)
        == []
    )
