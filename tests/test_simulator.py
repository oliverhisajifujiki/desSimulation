# ============================================================
# test_simulator.py
# ------------------------------------------------------------
# Tests for deterministic correctness of simulator equations.
# ============================================================

from src.simulator import (
    SimulationParams,
    ObservedState,
    computeConcernScore,
    generateObservations,
    initializePatient,
    simulateSinglePatientDeterministic,
    updateLatentState,
    updateTreatment,
)


def testGenerateObservationsDeterministic() -> None:
    """
    Test that the observed variables exactly match the
    deterministic observation equations when there is no noise.
    """

    # Create parameter object
    params = SimulationParams()

    # Create a fixed latent patient state
    patientState = initializePatient(
        severity=0.5,
        infection=0.4,
        respiratoryStress=0.3,
        advocacy=0.0,
    )

    # Generate observations from the latent state
    observedState = generateObservations(patientState, params)

    # Compute the expected values directly from the current
    # parameter object. This avoids tests breaking if parameters
    # are later tuned.
    expectedHeartRate = (
        params.alpha0
        + params.alpha1 * patientState.severity
        + params.alpha2 * patientState.infection
    )
    expectedRespiratoryRate = (
        params.beta0
        + params.beta1 * patientState.severity
        + params.beta2 * patientState.respiratoryStress
    )
    expectedTemperature = (
        params.delta0
        + params.delta1 * patientState.infection
    )

    # Verify exact equality in this deterministic setting
    assert observedState.heartRate == expectedHeartRate
    assert observedState.respiratoryRate == expectedRespiratoryRate
    assert observedState.temperature == expectedTemperature


def testConcernScoreAndTreatmentTrigger() -> None:
    """
    Test that:
    1. the concern score is computed correctly, and
    2. treatment activates when the score exceeds the threshold.
    """

    params = SimulationParams()

    # Construct an observed state with clearly elevated values
    observedState = ObservedState(
        heartRate=120.0,
        respiratoryRate=30.0,
        temperature=39.0,
    )

    # Latent state itself is irrelevant for this particular test,
    # since treatment is triggered from observed values
    patientState = initializePatient()

    # Compute concern score from the function
    concernScore = computeConcernScore(observedState, params)

    # Compute concern score manually from the current params
    expectedScore = (
        params.kappa1 * observedState.heartRate
        + params.kappa2 * observedState.respiratoryRate
        + params.kappa3 * observedState.temperature
    )

    # Verify the score is correct
    assert concernScore == expectedScore

    # Verify treatment turns on
    assert updateTreatment(patientState, observedState, params) == 1


def testUpdateLatentStateDeterministicUntreated() -> None:
    """
    Test one-step latent state update when treatment = 0.

    This verifies that the latent dynamics equations are being
    implemented correctly in the untreated case.
    """

    params = SimulationParams()

    # Fixed initial latent state
    patientState = initializePatient(
        severity=0.5,
        infection=0.4,
        respiratoryStress=0.3,
        advocacy=0.0,
    )
    patientState.treatment = 0

    # Update one time step
    nextState = updateLatentState(patientState, params)

    # Compute expected values directly from the equations
    expectedSeverity = (
        patientState.severity
        + params.a1 * patientState.infection
        + params.a2 * patientState.respiratoryStress
        - params.a3 * patientState.treatment
    )
    expectedInfection = (
        patientState.infection
        + params.b1
        - params.b2 * patientState.treatment
    )
    expectedRespiratoryStress = (
        patientState.respiratoryStress
        + params.c1 * patientState.severity
        + params.c2 * patientState.infection
        - params.c3 * patientState.treatment
    )

    # Verify exact match
    assert nextState.severity == expectedSeverity
    assert nextState.infection == expectedInfection
    assert nextState.respiratoryStress == expectedRespiratoryStress


def testUpdateLatentStateDeterministicTreated() -> None:
    """
    Test one-step latent state update when treatment = 1.

    This verifies that treatment enters the equations with the
    correct sign and magnitude.
    """

    params = SimulationParams()

    # Fixed initial latent state
    patientState = initializePatient(
        severity=0.5,
        infection=0.4,
        respiratoryStress=0.3,
        advocacy=0.0,
    )
    patientState.treatment = 1

    # Update one time step
    nextState = updateLatentState(patientState, params)

    # Compute expected values directly from the equations
    expectedSeverity = (
        patientState.severity
        + params.a1 * patientState.infection
        + params.a2 * patientState.respiratoryStress
        - params.a3 * patientState.treatment
    )
    expectedInfection = (
        patientState.infection
        + params.b1
        - params.b2 * patientState.treatment
    )
    expectedRespiratoryStress = (
        patientState.respiratoryStress
        + params.c1 * patientState.severity
        + params.c2 * patientState.infection
        - params.c3 * patientState.treatment
    )

    # Verify exact match
    assert nextState.severity == expectedSeverity
    assert nextState.infection == expectedInfection
    assert nextState.respiratoryStress == expectedRespiratoryStress


def testSimulateSinglePatientDeterministicLength() -> None:
    """
    Test that the deterministic simulator returns the correct
    number of time steps and labels the steps correctly.
    """

    params = SimulationParams()
    initialState = initializePatient()

    # Run a short 5-step rollout
    history = simulateSinglePatientDeterministic(
        initialState,
        params,
        numSteps=5,
    )

    # Check length
    assert len(history) == 5

    # Check first and last step index
    assert history[0]["step"] == 0
    assert history[-1]["step"] == 4