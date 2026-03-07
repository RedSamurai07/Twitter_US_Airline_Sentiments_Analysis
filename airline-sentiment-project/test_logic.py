import pytest
import re
from train import clean_text

def test_text_cleaning():
    # Test if links and special characters are removed
    sample = "Check this out http://test.com @user #awesome!"
    cleaned = clean_text(sample)
    assert "http" not in cleaned
    assert "@user" not in cleaned
    assert "#awesome" not in cleaned

def test_empty_string():
    assert clean_text("") == ""