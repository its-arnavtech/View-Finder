import numpy as np
import pytest

from viewfinder.world import locate
from viewfinder.world.scene import Scene


def test_regional_mass_is_the_share_near_the_best_guess():
    coordinates = np.array([[45.4, -80.3], [45.0, -79.5], [46.0, -81.0], [-33.9, 151.2]])
    probabilities = np.array([0.3, 0.25, 0.2, 0.25])
    best, mass = locate.regional_mass(probabilities, coordinates)
    assert best == 0
    assert mass == pytest.approx(0.75)  # Sydney is not near Ontario
    four_continents = np.array([[45.4, -80.3], [-33.9, 151.2], [51.5, -0.1], [35.7, 139.7]])
    _, spread = locate.regional_mass(np.full(4, 0.25), four_continents)
    assert spread == pytest.approx(0.25)  # no idea where


def test_location_breaks_a_close_call_but_never_overrules_the_photo():
    allowed = np.array([True, True, True, False])
    similarity = np.array([0.163, 0.207, 0.153, 0.9])
    # Ontario: "a mountain lake" 0.63 against "a calm lake among forested hills" 0.27.
    assert locate.rerank(np.array([0.63, 0.27, 0.06, 0.04]), similarity, 0.98, allowed) == 1
    # A clear call stays.
    assert locate.rerank(np.array([0.97, 0.02, 0.005, 0.005]), similarity, 0.98, allowed) == 0
    # Never another kind of setting, never something the photo does not show.
    assert locate.rerank(np.array([0.5, 0.0, 0.0, 0.5]), similarity, 1.0, allowed) == 0
    assert locate.rerank(np.array([0.95, 0.04, 0.01, 0.0]), np.array([0.0, 0.0, 5.0, 0.0]), 1.0, allowed) == 0
    # Untrusted (no confidence): the photo alone.
    assert locate.rerank(np.array([0.63, 0.27, 0.06, 0.04]), similarity, 0.0, allowed) == 0


def test_region_names_are_offline_and_readable():
    name = locate.region_name(45.39, -80.32)
    assert name.startswith("Ontario")


def test_the_prompt_names_the_country_around():
    scene = Scene(
        "a calm lake surrounded by forested hills", "nature", "on a sunny day", 0.9, "in summer",
        terrain=locate.TERRAINS[0], region="Ontario, Canada",
    )
    prompt = scene.prompt()
    assert "granite outcrops" in prompt and "in Ontario, Canada" in prompt
    assert "Ontario" in scene.prompt("down")
    assert scene.as_dict()["region"] == "Ontario, Canada"


def test_location_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(locate.LOCATE_ENV, "0")
    assert not locate.enabled()
    monkeypatch.delenv(locate.LOCATE_ENV)
    assert locate.enabled()
