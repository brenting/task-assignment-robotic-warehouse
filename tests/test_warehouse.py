import pytest

from tarware.warehouse import RewardType, Warehouse


@pytest.fixture
def env() -> Warehouse:
    ware_env = Warehouse(
        column_height=8,
        shelf_rows=3,
        shelf_columns=3,
        n_agvs=10,
        n_pickers=10,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=6,
        max_inactivity_steps=None,
        max_steps=500,
        reward_type=RewardType.INDIVIDUAL,
    )
    return ware_env


def test_smoke(env: Warehouse):
    env


def test_reset(env: Warehouse):
    env.reset()
