"""A gp_log.jsonl line cut by a kill (C22): read_gp_log and the resume."""
import json

import pytest
import torch

import bo

ROW1 = '{"iteration": 1, "stage": "acquisition"}'
ROW2 = '{"iteration": 2, "stage": "acquisition"}'
CUT = '{"iteration": 3, "stage": "acqui'


def _log(tmp_path, text):
    log = tmp_path / "gp_log"
    log.mkdir()
    (log / "gp_log.jsonl").write_bytes(text.encode())
    return log


def _resume(tmp_path, log, monkeypatch):
    """run_bo_resumable up to the point where it would call run_bo."""
    monkeypatch.setattr(bo, "run_bo", lambda **kw: (torch.zeros(1), [0.0]))
    ps = bo.ParamSpace({"n": {"bounds": (6.0, 12.0), "init": 8.0}})
    bo.run_bo_resumable(lambda *a, **k: None, ps, "", str(tmp_path), max_evals=1,
                        gp_log_dir=str(log))


def _moved(log):
    path = log / bo.GP_LOG_CUT_FILE
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines()]


def test_read_skips_an_unterminated_cut_last_line(tmp_path, capsys):
    log = _log(tmp_path, f"{ROW1}\n{ROW2}\n{CUT}")
    before = (log / "gp_log.jsonl").read_bytes()
    rows = bo.read_gp_log(log, raw=True)
    assert [r["iteration"] for r in rows] == [1, 2]
    assert "line 3 has no newline and is not valid JSON" in capsys.readouterr().out
    assert (log / "gp_log.jsonl").read_bytes() == before


def test_read_keeps_a_complete_last_line_without_newline(tmp_path):
    log = _log(tmp_path, f"{ROW1}\n{ROW2}")
    assert [r["iteration"] for r in bo.read_gp_log(log, raw=True)] == [1, 2]


@pytest.mark.parametrize("text", [
    f"{ROW1}\n{CUT}\n{ROW2}\n",       # bad line in the middle
    f"{ROW1}\n{CUT}\n",               # bad last line that has a newline
])
def test_read_raises_on_any_other_bad_line(tmp_path, text):
    log = _log(tmp_path, text)
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        bo.read_gp_log(log, raw=True)


def test_resume_moves_the_cut_last_line_aside(tmp_path, monkeypatch, capsys):
    log = _log(tmp_path, f"{ROW1}\n{ROW2}\n{CUT}")
    _resume(tmp_path, log, monkeypatch)

    assert (log / "gp_log.jsonl").read_text() == f"{ROW1}\n{ROW2}\n"
    [rec] = _moved(log)
    assert rec["text"] == CUT                       # exactly as written
    assert rec["line"] == 3
    assert rec["byte_offset"] == len(ROW1) + len(ROW2) + 2
    assert rec["n_bytes"] == len(CUT)
    assert rec["had_newline"] is False
    assert rec["source"] == "gp_log.jsonl"
    assert "moved to" in capsys.readouterr().out

    # The next row starts on its own line and the log reads cleanly.
    with open(log / "gp_log.jsonl", "a") as fh:
        fh.write('{"iteration": 3, "stage": "acquisition"}\n')
    assert [r["iteration"] for r in bo.read_gp_log(log, raw=True)] == [1, 2, 3]

    # A second resume finds nothing to move.
    _resume(tmp_path, log, monkeypatch)
    assert len(_moved(log)) == 1


def test_resume_moves_a_cut_line_that_an_older_resume_terminated(tmp_path, monkeypatch):
    # Before C22 a resume only added the newline after the cut line.
    log = _log(tmp_path, f"{ROW1}\n{CUT}\n")
    _resume(tmp_path, log, monkeypatch)
    assert (log / "gp_log.jsonl").read_text() == f"{ROW1}\n"
    [rec] = _moved(log)
    assert rec["text"] == CUT and rec["had_newline"] is True


def test_resume_keeps_a_complete_last_line_and_adds_its_newline(tmp_path, monkeypatch):
    log = _log(tmp_path, f"{ROW1}\n{ROW2}")
    _resume(tmp_path, log, monkeypatch)
    assert (log / "gp_log.jsonl").read_text() == f"{ROW1}\n{ROW2}\n"
    assert _moved(log) == []


def test_resume_raises_on_a_bad_middle_line_and_changes_nothing(tmp_path, monkeypatch):
    log = _log(tmp_path, f"{ROW1}\n{CUT}\n{ROW2}\n{CUT}")
    before = (log / "gp_log.jsonl").read_bytes()
    called = []
    monkeypatch.setattr(bo, "run_bo", lambda **kw: called.append(kw))
    ps = bo.ParamSpace({"n": {"bounds": (6.0, 12.0), "init": 8.0}})
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        bo.run_bo_resumable(lambda *a, **k: None, ps, "", str(tmp_path), max_evals=1,
                            gp_log_dir=str(log))
    assert (log / "gp_log.jsonl").read_bytes() == before
    assert _moved(log) == [] and called == []


def test_resume_without_a_gp_log(tmp_path, monkeypatch):
    log = tmp_path / "gp_log"
    _resume(tmp_path, log, monkeypatch)
    assert not (log / bo.GP_LOG_CUT_FILE).exists()
