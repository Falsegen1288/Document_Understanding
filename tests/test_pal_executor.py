import pytest
from src.arithmetic.pal_executor import run_pal_code, extract_code_block

def test_pal_yoy():
    r = run_pal_code("result = (15890.0 - 14205.4) / 14205.4 * 100")
    assert r.success is True
    assert abs(r.value - 11.8589) < 0.01

def test_pal_cagr():
    r = run_pal_code("result = ((15890.0 / 12000.0) ** (1/3) - 1) * 100")
    assert r.success is True
    assert abs(r.value - 9.8114) < 0.01

def test_adversarial_import():
    r = run_pal_code('import os\nresult = len(os.listdir("."))')
    assert r.success is False
    assert "Import" in r.error

def test_adversarial_eval():
    r = run_pal_code('result = eval("1+1")')
    assert r.success is False
    assert "eval" in r.error

def test_adversarial_infinite_loop():
    r = run_pal_code("while True: pass", timeout_s=1.5)
    assert r.success is False
    assert "timeout" in r.error

def test_malformed_missing_result():
    r = run_pal_code("x = 5 + 5")
    assert r.success is False
    assert "result" in r.error

def test_extract_code_block():
    llm_out = "Here is the code:\n```python\nresult = 42 * 2\n```\nDone."
    code = extract_code_block(llm_out)
    assert code == "result = 42 * 2"
