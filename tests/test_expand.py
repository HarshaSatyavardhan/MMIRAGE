"""Tests for the expand output feature (Issue #11)."""

import pytest
from mmirage.core.process.variables import VariableEnvironment
from mmirage.core.process.mapper import MMIRAGEMapper


class TestExpandEnvironments:
    """Test _expand_environments logic in MMIRAGEMapper."""

    def _make_env(self, **kwargs):
        return VariableEnvironment(kwargs)

    def _make_mapper_with_expand(self, expand=True):
        """Create a mapper with a mock expand output var (no real processor)."""
        from mmirage.core.process.processors.llm.config import LLMOutputVar

        output_var = LLMOutputVar(
            name="qa_pairs",
            type="llm",
            prompt="dummy",
            output_type="JSON",
            output_schema=["question", "answer"],
            expand=expand,
        )
        # We only need the mapper for _expand_environments, not actual processing
        mapper = MMIRAGEMapper.__new__(MMIRAGEMapper)
        mapper.processors = {}
        mapper.input_vars = []
        mapper.output_vars = [output_var]
        return mapper

    def test_expand_list_creates_multiple_envs(self):
        """A list of 3 items should produce 3 environments."""
        mapper = self._make_mapper_with_expand(expand=True)
        env = self._make_env(
            text="hello",
            qa_pairs=[
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
                {"question": "Q3", "answer": "A3"},
            ],
        )

        result = mapper._expand_environments([env])
        assert len(result) == 3
        assert result[0].get("qa_pairs") == {"question": "Q1", "answer": "A1"}
        assert result[1].get("qa_pairs") == {"question": "Q2", "answer": "A2"}
        assert result[2].get("qa_pairs") == {"question": "Q3", "answer": "A3"}
        # Other vars preserved
        for r in result:
            assert r.get("text") == "hello"

    def test_expand_single_item_list(self):
        """A list with 1 item should produce 1 environment."""
        mapper = self._make_mapper_with_expand(expand=True)
        env = self._make_env(
            text="hello",
            qa_pairs=[{"question": "Q1", "answer": "A1"}],
        )

        result = mapper._expand_environments([env])
        assert len(result) == 1
        assert result[0].get("qa_pairs") == {"question": "Q1", "answer": "A1"}

    def test_expand_empty_list_keeps_env(self):
        """An empty list should keep the original environment."""
        mapper = self._make_mapper_with_expand(expand=True)
        env = self._make_env(text="hello", qa_pairs=[])

        result = mapper._expand_environments([env])
        assert len(result) == 1
        assert result[0].get("qa_pairs") == []

    def test_expand_non_list_keeps_env(self):
        """A non-list value should keep the original environment."""
        mapper = self._make_mapper_with_expand(expand=True)
        env = self._make_env(text="hello", qa_pairs={"question": "Q1", "answer": "A1"})

        result = mapper._expand_environments([env])
        assert len(result) == 1

    def test_no_expand_flag_passthrough(self):
        """When expand=False, environments pass through unchanged."""
        mapper = self._make_mapper_with_expand(expand=False)
        env = self._make_env(
            text="hello",
            qa_pairs=[
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ],
        )

        result = mapper._expand_environments([env])
        assert len(result) == 1
        assert isinstance(result[0].get("qa_pairs"), list)

    def test_expand_multiple_input_rows(self):
        """Multiple input rows, each with different list lengths."""
        mapper = self._make_mapper_with_expand(expand=True)
        envs = [
            self._make_env(text="row1", qa_pairs=[
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ]),
            self._make_env(text="row2", qa_pairs=[
                {"question": "Q3", "answer": "A3"},
            ]),
            self._make_env(text="row3", qa_pairs=[
                {"question": "Q4", "answer": "A4"},
                {"question": "Q5", "answer": "A5"},
                {"question": "Q6", "answer": "A6"},
            ]),
        ]

        result = mapper._expand_environments(envs)
        assert len(result) == 6  # 2 + 1 + 3
        assert result[0].get("text") == "row1"
        assert result[0].get("qa_pairs") == {"question": "Q1", "answer": "A1"}
        assert result[2].get("text") == "row2"
        assert result[3].get("text") == "row3"
