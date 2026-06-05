import pytest
from app import app, clean_text

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client: yield client

def test_clean_text_basic():
    assert clean_text("Hello @user! http://test.com") == "hello"

def test_clean_text_non_string():
    assert clean_text(None) == ""

def test_health(client):
    assert client.get('/health').status_code == 200

def test_predict_success(client):
    res = client.post('/predict', json={'tweet': 'I love this!'})
    assert res.status_code == 200
    assert 'sentiment' in res.json

def test_predict_failure(client):
    # Testing branch coverage for missing data
    assert client.post('/predict', json={}).status_code == 400