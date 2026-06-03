import pytest

from vlmparse.build_doc import parse_page_selection
from vlmparse.converter_with_server import ConverterWithServer


class TestParsePageSelection:
    """Unit tests for the 1-based page-selection string parser."""

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("1", [0]),
            ("3", [2]),
            ("1,2,3", [0, 1, 2]),
            ("1-5", [0, 1, 2, 3, 4]),
            ("1-5,8,10", [0, 1, 2, 3, 4, 7, 9]),
            ("1-1", [0]),  # single-page range
            ("2,2,2", [1]),  # dedupe
            ("10,1,5", [0, 4, 9]),  # sorted
            (" 1 , 3 ", [0, 2]),  # whitespace tolerant
        ],
    )
    def test_valid_specs(self, spec, expected):
        assert parse_page_selection(spec) == expected

    def test_num_pages_drops_out_of_range(self):
        assert parse_page_selection("1-10", num_pages=3) == [0, 1, 2]

    @pytest.mark.parametrize("spec", ["0", "1-0", "5-2", "abc", "1-x", "-1"])
    def test_invalid_specs_raise(self, spec):
        with pytest.raises(ValueError):
            parse_page_selection(spec)


class TestParseWithPageFilter:
    """Integration test: parse only a subset of pages."""

    def test_parse_subset_of_pages(
        self, mock_docker_operations, datadir, mock_openai_api, tmp_path
    ):
        test_file = datadir / "Fiche_Graines_A5.pdf"

        with mock_docker_operations(model_filter=lambda model: False):
            with mock_openai_api():
                with ConverterWithServer(model="gemini-2.5-flash-lite") as parser:
                    parser.client.return_documents_in_batch_mode = True
                    documents = parser.parse(
                        inputs=[str(test_file)],
                        out_folder=str(tmp_path),
                        mode="md",
                        debug=True,
                        pages=[0],
                    )

        assert documents is not None
        assert len(documents) == 1
        assert len(documents[0].pages) == 1
        assert documents[0].pages[0].page_number == 0
