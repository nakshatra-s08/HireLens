"""
Unit tests for Privacy & Anti-Bias Scrubber.
"""

import pytest
from src.privacy_scrubber import scrub_demographics

def test_demographic_scrubbing():
    raw_text = "My name is John Doe, 28 years old, male, living in San Francisco. I graduated from Stanford University. I have 5 years Python experience."
    clean_text, redactions = scrub_demographics(raw_text)

    assert "John Doe" not in clean_text
    assert "28 years old" not in clean_text
    assert "male" not in clean_text
    assert "Stanford University" not in clean_text
    assert "San Francisco" not in clean_text
    assert "Python" in clean_text
    assert len(redactions) >= 4

def test_gender_and_age_scrubbing():
    text = "She is 25 years old and born in 1999."
    clean, red = scrub_demographics(text)
    assert "25 years old" not in clean
    assert "1999" not in clean

if __name__ == "__main__":
    test_demographic_scrubbing()
    test_gender_and_age_scrubbing()
    print("Privacy tests passed!")
