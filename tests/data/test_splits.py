from src.data import splits


def test_split_disjoint_and_proportional(tiny_atc_map):
    ingredients = ["ibuprofen", "naproxen", "metformin"]
    train, test = splits.stratified_split(ingredients, tiny_atc_map,
                                          test_fraction=0.34, seed=42)
    assert set(train).isdisjoint(set(test))
    assert set(train) | set(test) == set(ingredients)
    assert len(test) == 1  # 34% of 3, rounded


def test_split_is_deterministic_across_seeds(tiny_atc_map):
    ingredients = ["ibuprofen", "naproxen", "metformin"]
    a = splits.stratified_split(ingredients, tiny_atc_map, 0.34, seed=42)
    b = splits.stratified_split(ingredients, tiny_atc_map, 0.34, seed=42)
    assert a == b


def test_split_unmapped_ingredients_go_to_a_random_bucket(tiny_atc_map):
    ingredients = ["ibuprofen", "naproxen", "metformin", "unknown_x", "unknown_y"]
    atc_map = {**tiny_atc_map, "unknown_x": [], "unknown_y": []}
    train, test = splits.stratified_split(ingredients, atc_map, 0.4, seed=42)
    assert set(train).isdisjoint(set(test))
    assert "unknown_x" in train + test
    assert "unknown_y" in train + test
    # The two unknowns share the __UNMAPPED__ stratum; round(2 * 0.4) = 1
    # of them must end up in test (proves the stratum was actually split).
    unknowns_in_test = [x for x in ("unknown_x", "unknown_y") if x in test]
    assert len(unknowns_in_test) == 1


def test_save_and_load_round_trip(tmp_path):
    train = ["ibuprofen", "metformin"]
    test = ["naproxen"]
    train_path = tmp_path / "train.txt"
    test_path  = tmp_path / "test.txt"
    splits.save_split(train, test, train_path, test_path)
    loaded_train, loaded_test = splits.load_split(train_path, test_path)
    assert loaded_train == train
    assert loaded_test == test
