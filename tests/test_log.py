from blocks.log import TrainingLog


def test_training_log():
    log = TrainingLog()

    # test basic writing capabilities
    log[0].field = 45
    assert log[0].field == 45
    assert log[1].field is None
    assert log.current_row.field == 45
    log.status.iterations_done += 1
    assert log.status.iterations_done == 1
    assert log.previous_row.field == 45

    # test default values mechanism
    log.set_default_value('flag', True)
    assert log[0].flag
    log[1].flag = False
    assert not log[1].flag

    # test iteration
    assert len(list(log)) == 2
    df = log.to_dataframe()
    assert list(sorted(df.columns)) == ["field", "flag"]
    assert df.flag[1] is False
    assert df.field[0] == 45
