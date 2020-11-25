import pytest

from quickspacer import Spacer


@pytest.fixture
def model1():
    return Spacer(level=1)


@pytest.fixture
def model2():
    return Spacer(level=2)


def test_batch_space(model1):
    texts = ["안녕하세요", "이것좀띄워보게나자네분위기도좀띄우고말야", "지금제말이무슨말인지아시겠죠?", "AI인공지능만세!!!"]
    assert model1.space(texts) == model1.space(texts, batch_size=2)


def test_empty_space(model2):
    assert model2.space([]) == []
