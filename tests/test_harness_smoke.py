import pytest
from benchmark_harness.config import PipelineConfig
from benchmark_harness.runner import BenchmarkRunner


def test_benchmark_harness_smoke_tatdqa():
    config = PipelineConfig(pipeline="baseline", dataset="tatdqa", limit=3)
    runner = BenchmarkRunner(config)
    result = runner.run()

    assert isinstance(result, dict)
    assert result["pipeline"] == "baseline"
    assert result["dataset"] == "tatdqa"
    assert result["num_queries"] <= 3
    assert "metrics" in result
    assert "oracle_scoped" in result["metrics"]
    assert "true_e2e" in result["metrics"]
    assert "sanity_check" in result
    assert result["sanity_check"]["true_e2e_le_oracle"] is True


def test_benchmark_harness_smoke_unidoc():
    config = PipelineConfig(pipeline="baseline", dataset="unidoc", limit=3)
    runner = BenchmarkRunner(config)
    result = runner.run()

    assert isinstance(result, dict)
    assert result["pipeline"] == "baseline"
    assert result["dataset"] == "unidoc"
    assert result["num_queries"] <= 3
    assert result["sanity_check"]["true_e2e_le_oracle"] is True
