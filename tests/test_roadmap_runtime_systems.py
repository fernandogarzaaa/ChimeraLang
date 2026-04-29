def test_causal_model_records_adjustment_strategy():
    from chimera_runtime import CausalModel

    model = CausalModel(name="ChimeraCausal", adjustment="backdoor")

    result = model.causal_effect("x", adjust_for=["z"])

    assert result["model"] == "ChimeraCausal"
    assert result["adjustment"] == "backdoor"
    assert result["adjust_for"] == ["z"]


def test_differential_privacy_engine_clips_and_tracks_budget():
    from chimera_runtime import DifferentialPrivacyEngine

    engine = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, clip_norm=1.0, seed=1)

    noisy = engine.privatize([2.0, 0.0])

    assert len(noisy) == 2
    assert engine.consumed_epsilon == 1.0
    assert engine.remaining_budget == 0.0


def test_meta_learner_and_self_improver_return_auditable_records():
    from chimera_runtime import MetaLearner, SelfImprover

    meta = MetaLearner(base_model="ChimeraMoE", inner_steps=5)
    adapted = meta.adapt(new_task_examples=5)
    improver = SelfImprover(model="AlignedChimera", verifier="z3_solver")
    proposal = improver.propose({"lr": 0.001})

    assert adapted["inner_steps"] == 5
    assert proposal["verified"] is True
    assert proposal["verifier"] == "z3_solver"


def test_biological_runtime_primitives_record_state():
    from chimera_runtime import PredictiveCodingRuntime, ReplayBuffer, RewardSystem

    replay = ReplayBuffer(capacity=2, strategy="prioritized")
    replay.add("old", priority=0.1)
    replay.add("surprising", priority=0.9)
    reward = RewardSystem().prediction_error(expected=0.2, actual=0.9)
    predictive = PredictiveCodingRuntime(depth=2).propagate([0.0, 1.0])

    assert replay.sample(1) == ["surprising"]
    assert reward == 0.7
    assert predictive["layers_evaluated"] == 2
