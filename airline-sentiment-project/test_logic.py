import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from app import app, clean_text

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- clean_text tests ---

def test_clean_text_basic():
    assert clean_text("Hello @user! http://test.com") == "hello"

def test_clean_text_non_string():
    assert clean_text(None) == ""

def test_clean_text_removes_hashtag():
    result = clean_text("great flight #awesome")
    assert "#awesome" not in result

# --- Route tests ---

def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json['status'] == 'healthy'

def test_predict_missing_tweet_key(client):
    assert client.post('/predict', json={}).status_code == 400

def test_predict_no_body(client):
    resp = client.post('/predict', content_type='application/json', data='{}')
    assert resp.status_code == 400

@patch('app.get_resources')
def test_predict_success(client, mock_res):   # ← fixture first, mock second
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    mock_tokenizer = MagicMock()
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ['positive']
    mock_res.return_value = (mock_model, mock_tokenizer, mock_le, set())

    res = client.post('/predict', json={'tweet': 'I love this airline!'})
    assert res.status_code == 200
    assert res.json['sentiment'] == 'positive'

@patch('app.get_resources')
def test_predict_negative_sentiment(client, mock_res):   # ← fixture first, mock second
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
    mock_tokenizer = MagicMock()
    mock_tokenizer.texts_to_sequences.return_value = [[5, 6]]
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ['negative']
    mock_res.return_value = (mock_model, mock_tokenizer, mock_le, set())

    res = client.post('/predict', json={'tweet': 'Terrible flight, never again'})
    assert res.status_code == 200
    assert res.json['sentiment'] == 'negative'