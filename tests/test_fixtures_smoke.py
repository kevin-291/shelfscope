from __future__ import annotations


def test_conftest_fixtures_are_loaded(client, fake_services, uploaded_image):
    assert client is not None
    assert fake_services["detector"] is not None
    assert uploaded_image[0] == "sample.png"
