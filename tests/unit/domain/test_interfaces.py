from battle_royale.domain.interfaces.environment import IBattleRoyaleEnv
from battle_royale.domain.interfaces.logger import ILogger
from battle_royale.domain.interfaces.policy import IPolicy


def test_environment_protocol_is_protocol():
    # Protocols should be importable and have the required method names
    assert hasattr(IBattleRoyaleEnv, "reset")
    assert hasattr(IBattleRoyaleEnv, "step")
    assert hasattr(IBattleRoyaleEnv, "get_agents")


def test_logger_protocol_has_required_methods():
    assert hasattr(ILogger, "log")
    assert hasattr(ILogger, "save_artifact")


def test_policy_protocol_has_required_methods():
    assert hasattr(IPolicy, "predict")
    assert hasattr(IPolicy, "save")