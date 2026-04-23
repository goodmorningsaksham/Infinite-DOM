"""Phase 5 tests — a11y tree formatter (no browser required)."""
from infinite_dom.browser.a11y_formatter import format_a11y_tree


def test_empty_snapshot():
    r = format_a11y_tree(None)
    assert "empty" in r.text.lower()


def test_single_button():
    snap = {
        "role": "WebArea",
        "name": "",
        "children": [
            {"role": "button", "name": "Search", "children": []},
        ],
    }
    r = format_a11y_tree(snap)
    assert "btn_1" in r.text
    assert "Search" in r.text
    assert "btn_1" in r.ref_map


def test_multiple_buttons_get_unique_refs():
    snap = {
        "role": "main",
        "name": "",
        "children": [
            {"role": "button", "name": "Click A"},
            {"role": "button", "name": "Click B"},
        ],
    }
    r = format_a11y_tree(snap)
    assert "btn_1" in r.ref_map
    assert "btn_2" in r.ref_map
    assert r.ref_map["btn_1"].name == "Click A"
    assert r.ref_map["btn_2"].name == "Click B"


def test_indentation_reflects_depth():
    snap = {
        "role": "main",
        "children": [
            {"role": "form", "children": [
                {"role": "button", "name": "Submit"},
            ]},
        ],
    }
    r = format_a11y_tree(snap)
    lines = r.text.split("\n")
    btn_line = next((line for line in lines if "btn_" in line), "")
    assert btn_line.startswith("  ")


def test_form_elements_distinguished():
    snap = {
        "role": "form",
        "children": [
            {"role": "textbox", "name": "Email"},
            {"role": "combobox", "name": "Country"},
        ],
    }
    r = format_a11y_tree(snap)
    assert "inp_1" in r.ref_map
    assert "cmb_1" in r.ref_map
