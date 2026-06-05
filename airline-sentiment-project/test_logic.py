import pytest
from app import app, clean_text

# 1. Test the utility function (Logic layer)
def test_text_cleaning():
    sample = "Check this out http://test.com @user #awesome!"
    cleaned = clean_text(sample)
    assert "http" not in cleaned
    assert "@user" not in cleaned
    assert "#awesome" not in cleaned

def test_empty_string():
    assert clean_text("") == ""

# 2. Test the API routes (Application layer)
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_route(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_predict_route_success(client):
    # Testing the /predict endpoint
    payload = {'tweet': 'I love flying with this airline!'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert 'sentiment' in response.json

def test_predict_route_failure(client):
    # Testing error handling when no data is provided
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert 'error' in response.json