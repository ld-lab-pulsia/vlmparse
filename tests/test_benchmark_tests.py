import pytest
from pydantic import ValidationError

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import (
    BaselineTest,
    BasePDFTest,
    TextOrderTest,
    TextPresenceTest,
    normalize_text,
)


class TestNormalizeText:
    """Test the normalize_text function"""

    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized"""
        input_text = "This  has\tmultiple    spaces\nand\nnewlines"
        expected = "This has multiple spaces and newlines"
        assert normalize_text(input_text) == expected

    def test_character_replacement(self):
        """Test that fancy characters are replaced with ASCII equivalents"""
        input_text = "This has 'fancy' \u201cquotes\u201d and\u2014dashes"
        expected = "This has 'fancy' \"quotes\" and-dashes"
        assert normalize_text(input_text) == expected

    def test_markdown1(self):
        """Test that fancy characters are replaced with ASCII equivalents"""
        input_text = "this is *bold*"
        expected = "this is bold"
        assert normalize_text(input_text) == expected

    def test_markdown2(self):
        """Test that fancy characters are replaced with ASCII equivalents"""
        input_text = "_italic__ is *bold*"
        expected = "italic_ is bold"
        assert normalize_text(input_text) == expected

    def test_empty_input(self):
        """Test that empty input returns empty output"""
        assert normalize_text("") == ""

    def test_brs(self):
        """Test that empty input returns empty output"""
        assert normalize_text("Hello<br>everyone") == "Hello everyone"
        assert normalize_text("Hello<br>everyone") == normalize_text("Hello\neveryone")
        assert normalize_text("Hello<br/>everyone") == "Hello everyone"
        assert normalize_text("Hello<br/>everyone") == normalize_text("Hello\neveryone")

    def test_two_stars(self):
        assert (
            normalize_text(
                "**Georges V.** (2007) – *Le Forez du VIe au IVe millénaire av. J.-C. Territoires, identités et stratégies des sociétés humaines du Massif central dans le bassin amont de la Loire (France)*, thèse de doctorat, université de Bourgogne, Dijon, 2 vol., 435 p."
            )
            == "Georges V. (2007) - Le Forez du VIe au IVe millénaire av. J.-C. Territoires, identités et stratégies des sociétés humaines du Massif central dans le bassin amont de la Loire (France), thèse de doctorat, université de Bourgogne, Dijon, 2 vol., 435 p."
        )


class TestBasePDFTest:
    """Test the BasePDFTest class"""

    def test_valid_initialization(self):
        """Test that a valid initialization works"""
        test = BasePDFTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        assert test.pdf == "test.pdf"
        assert test.page == 1
        assert test.id == "test_id"
        assert test.type == "baseline"
        assert test.max_diffs == 0
        assert test.checked is None
        assert test.url is None

    def test_empty_pdf(self):
        """Test that empty PDF raises ValidationError"""
        with pytest.raises(ValidationError):
            BasePDFTest(pdf="", page=1, id="test_id", type="baseline")

    def test_empty_id(self):
        """Test that empty ID raises ValidationError"""
        with pytest.raises(ValidationError):
            BasePDFTest(pdf="test.pdf", page=1, id="", type="baseline")

    def test_negative_max_diffs(self):
        """Test that negative max_diffs raises ValidationError"""
        with pytest.raises(ValidationError):
            BasePDFTest(
                pdf="test.pdf", page=1, id="test_id", type="baseline", max_diffs=-1
            )

    def test_invalid_test_type(self):
        """Test that invalid test type raises ValidationError"""
        with pytest.raises(ValidationError):
            BasePDFTest(pdf="test.pdf", page=1, id="test_id", type="invalid_type")

    def test_run_method_not_implemented(self):
        """Test that run method raises NotImplementedError"""
        test = BasePDFTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        with pytest.raises(NotImplementedError):
            test.run("content")

    def test_checked_enum(self):
        """Test that checked accepts valid TestChecked enums"""
        test = BasePDFTest(
            pdf="test.pdf", page=1, id="test_id", type="baseline", checked="verified"
        )
        assert test.checked == "verified"


class TestTextPresenceTest:
    """Test the TextPresenceTest class"""

    def test_valid_present_test(self):
        """Test that a valid PRESENT test initializes correctly"""
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="present", text="test text"
        )
        assert test.text == "test text"
        assert test.case_sensitive is True
        assert test.first_n is None
        assert test.last_n is None

    def test_valid_absent_test(self):
        """Test that a valid ABSENT test initializes correctly"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="absent",
            text="test text",
            case_sensitive=False,
        )
        assert test.text == "test text"
        assert test.case_sensitive is False

    def test_empty_text(self):
        """Test that empty text is allowed (no Pydantic validation on text length)"""
        # Note: The current implementation allows empty text - no validation constraint exists
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="present", text=""
        )
        assert test.text == ""

    def test_present_text_exact_match(self):
        """Test that PRESENT test returns True for exact match"""
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="present", text="target text"
        )
        result, _ = test.run("This is some target text in a document")
        assert result is True

    def test_present_text_not_found(self):
        """Test that PRESENT test returns False when text not found"""
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="present", text="missing text"
        )
        result, explanation = test.run("This document doesn't have the target")
        assert result is False
        assert "missing text" in explanation

    def test_present_text_with_max_diffs(self):
        """Test that PRESENT test with max_diffs handles fuzzy matching"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="present",
            text="target text",
            max_diffs=2,
        )
        result, _ = test.run("This is some targett textt in a document")
        assert result is True

    def test_absent_text_found(self):
        """Test that ABSENT test returns False when text is found"""
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="absent", text="target text"
        )
        result, explanation = test.run("This is some target text in a document")
        assert result is False
        assert "target text" in explanation

    def test_absent_text_found_diffs(self):
        """Test that ABSENT test with max_diffs handles fuzzy matching"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="absent",
            text="target text",
            max_diffs=2,
        )
        result, explanation = test.run("This is some target text in a document")
        assert result is False
        result, explanation = test.run("This is some targett text in a document")
        assert result is False
        # With max_diffs=2 and text "target text" (11 chars), threshold is ~0.82
        # "targettt text" still matches within fuzzy threshold
        result, explanation = test.run("This is some targettt text in a document")
        assert result is False
        # Even more diffs still matches due to fuzzy matching
        result, explanation = test.run("This is some targetttt text in a document")
        assert result is False

    def test_absent_text_not_found(self):
        """Test that ABSENT test returns True when text is not found"""
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="absent", text="missing text"
        )
        result, _ = test.run("This document doesn't have the target")
        assert result is True

    def test_case_insensitive_present(self):
        """Test that case_sensitive=False works for PRESENT test"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="present",
            text="TARGET TEXT",
            case_sensitive=False,
        )
        result, _ = test.run("This is some target text in a document")
        assert result is True

    def test_case_insensitive_absent(self):
        """Test that case_sensitive=False works for ABSENT test"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="absent",
            text="TARGET TEXT",
            case_sensitive=False,
        )
        result, explanation = test.run("This is some target text in a document")
        assert result is False

    def test_first_n_limit(self):
        """Test that first_n parameter works correctly"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="present",
            text="beginning",
            first_n=20,
        )
        result, _ = test.run("beginning of text, but not the end")
        assert result is True

        # Test that text beyond first_n isn't matched
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="present", text="end", first_n=20
        )
        result, _ = test.run("beginning of text, but not the end")
        assert result is False

    def test_last_n_limit(self):
        """Test that last_n parameter works correctly"""
        test = TextPresenceTest(
            pdf="test.pdf", page=1, id="test_id", type="present", text="end", last_n=20
        )
        result, _ = test.run("beginning of text, but not the end")
        assert result is True

        # Test that text beyond last_n isn't matched
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="present",
            text="beginning",
            last_n=20,
        )
        result, _ = test.run("beginning of text, but not the end")
        assert result is False

    def test_both_first_and_last_n(self):
        """Test that combining first_n and last_n works correctly"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="present",
            text="beginning",
            first_n=15,
            last_n=10,
        )
        result, _ = test.run("beginning of text, middle part, but not the end")
        assert result is True

        # Text only in middle shouldn't be found
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="present",
            text="middle",
            first_n=15,
            last_n=10,
        )
        result, _ = test.run("beginning of text, middle part, but not the end")
        assert result is False

    def test_unicode_normalized_forms(self):
        """Test that e+accent == e_with_accent unicode chars"""
        test = TextPresenceTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="present",
            text="I like to eat at a café",
        )
        result, _ = test.run("I like to eat at a café")
        assert result is True

        result, _ = test.run("I like to eat at a cafe\u0301")
        assert result is True


class TestTextOrderTest:
    """Test the TextOrderTest class"""

    def test_valid_initialization(self):
        """Test that valid initialization works"""
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="first text",
            after="second text",
        )
        assert test.before == "first text"
        assert test.after == "second text"

    def test_invalid_test_type(self):
        """Test that invalid test type raises ValidationError"""
        with pytest.raises(ValidationError):
            TextOrderTest(
                pdf="test.pdf",
                page=1,
                id="test_id",
                type="present",
                before="first text",
                after="second text",
            )

    def test_empty_before(self):
        """Test that empty before text is allowed (no Pydantic validation on text length)"""
        # Note: The current implementation allows empty text - no validation constraint exists
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="",
            after="second text",
        )
        assert test.before == ""

    def test_empty_after(self):
        """Test that empty after text is allowed (no Pydantic validation on text length)"""
        # Note: The current implementation allows empty text - no validation constraint exists
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="first text",
            after="",
        )
        assert test.after == ""

    def test_correct_order(self):
        """Test that correct order returns True"""
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="first",
            after="second",
        )
        result, _ = test.run("This has first and then second in correct order")
        assert result is True

    def test_incorrect_order(self):
        """Test that incorrect order returns False"""
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="second",
            after="first",
        )
        result, explanation = test.run(
            "This has first and then second in correct order"
        )
        assert result is False

    def test_before_not_found(self):
        """Test that 'before' text not found returns False"""
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="missing",
            after="present",
        )
        result, explanation = test.run("This text has present but not the other word")
        assert result is False

    def test_after_not_found(self):
        """Test that 'after' text not found returns False"""
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="present",
            after="missing",
        )
        result, explanation = test.run("This text has present but not the other word")
        assert result is False

    def test_max_diffs(self):
        """Test that max_diffs parameter works correctly"""
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="first",
            after="second",
            max_diffs=1,
        )
        result, _ = test.run("This has firsst and then secand in correct order")
        assert result is True

    def test_multiple_occurrences(self):
        """Test that multiple occurrences are handled correctly"""
        # When before == after, the test looks for the same text appearing twice
        # The logic finds all occurrences and checks if any before_match appears before any after_match
        # This will fail when they are the same text since no occurrence can be before itself
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="target",
            after="target",
        )
        result, explanation = test.run("This has target and then target again")
        # This should fail because "target" cannot appear before itself
        assert result is False

        # Test that when different texts appear in order, it works
        # Note: normalization removes spaces, so "first then second" becomes "firstthensecond"
        test = TextOrderTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type="order",
            before="first",
            after="second",
        )
        result, _ = test.run("first then second appears here")
        assert result is True


class TestBaselineTest:
    """Test the BaselineTest class"""

    def test_valid_initialization(self):
        """Test that valid initialization works"""
        test = BaselineTest(
            pdf="test.pdf", page=1, id="test_id", type="baseline", max_repeats=50
        )
        assert test.max_repeats == 50

    def test_non_empty_content(self):
        """Test that non-empty content passes"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        result, _ = test.run("This is some normal content")
        assert result is True

    def test_empty_content(self):
        """Test that empty content fails"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        result, explanation = test.run("   \n\t  ")
        assert result is False
        assert "no alpha numeric characters" in explanation

    def test_repeating_content(self):
        """Test that highly repeating content fails"""
        test = BaselineTest(
            pdf="test.pdf", page=1, id="test_id", type="baseline", max_repeats=2
        )
        # Create highly repeating content - repeat "abc" many times
        repeating_content = "abc" * 10
        result, explanation = test.run(repeating_content)
        assert result is False
        assert "repeating" in explanation

    def test_content_with_disallowed_characters(self):
        """Test that content with disallowed characters fails"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        result, explanation = test.run("This has Chinese characters: 你好")
        assert result is False
        assert "disallowed characters" in explanation

    def test_content_with_emoji(self):
        """Test that content with emoji fails"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        result, explanation = test.run("This has emoji: 😊")
        assert result is False
        assert "disallowed characters" in explanation
        assert "😊" in explanation

    def test_content_with_mandarin(self):
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        result, explanation = test.run("asdfasdfas維基百科/中文asdfw")
        assert result is False
        assert "disallowed characters" in explanation

    def test_valid_content(self):
        """Test that valid content passes all checks"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type="baseline")
        content = "This is some normal content with proper English letters and no suspicious repetition."
        result, _ = test.run(content)
        assert result is True
