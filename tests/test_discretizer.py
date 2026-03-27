from src.discretizer import StandardDiscretizer


def test_standard_discretizer_has_17_channels():
    d = StandardDiscretizer()
    assert len(d._id_to_channel) == 17


def test_standard_discretizer_impute_strategies():
    for strategy in ['zero', 'normal_value', 'previous', 'next']:
        d = StandardDiscretizer(impute_strategy=strategy)
        assert d._impute_strategy == strategy
