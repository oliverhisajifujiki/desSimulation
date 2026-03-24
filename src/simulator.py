# ============================================================
# simulator.py
# ------------------------------------------------------------
# This file defines a very simple deterministic simulator for:
# - latent patient state evolution
# - observed vitals generation
# - treatment triggering
# - short rollout (5 steps)
#
# IMPORTANT:
# - This version has NO randomness
# - It is purely to verify equations are implemented correctly
# ============================================================

from dataclasses import dataclass
from typing import List


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PatientState:
    """
    Represents the true (hidden) state of a patient at time t.

    severity: overall illness severity
    infection: infection / inflammatory burden
    respiratoryStress: respiratory load / difficulty
    advocacy: patient-level trait (how much they push for care)
    treatment: whether treatment has started (0 or 1)
    """
    severity: float
    infection: float
    respiratoryStress: float
    advocacy: float
    treatment: int = 0


@dataclass
class ObservedState:
    """
    Represents observed clinical measurements at time t.

    These are generated from the latent state.
    """
    heartRate: float
    respiratoryRate: float
    temperature: float


@dataclass
class SimulationParams:
    """
    Contains all parameters for the simulation.

    These include:
    - latent dynamics coefficients
    - observation equations
    - treatment threshold
    - clipping bounds
    """

    # --------------------------
    # LATENT DYNAMICS PARAMETERS
    # --------------------------
    a1: float = 0.08  # infection -> severity
    a2: float = 0.05  # respiratory stress -> severity
    a3: float = 0.08  # treatment reduces severity

    b1: float = 0.05  # baseline infection growth
    b2: float = 0.10  # treatment reduces infection

    c1: float = 0.06  # severity -> respiratory stress
    c2: float = 0.05  # infection -> respiratory stress
    c3: float = 0.10  # treatment reduces respiratory stress

    # --------------------------
    # OBSERVATION EQUATIONS
    # --------------------------
    alpha0: float = 80.0   # HR baseline
    alpha1: float = 18.0   # severity -> HR
    alpha2: float = 12.0   # infection -> HR

    beta0: float = 16.0    # RR baseline
    beta1: float = 4.0     # severity -> RR
    beta2: float = 6.0     # respiratory stress -> RR

    delta0: float = 36.8   # temperature baseline
    delta1: float = 1.2    # infection -> temperature

    # --------------------------
    # TREATMENT TRIGGER
    # --------------------------
    kappa1: float = 0.02   # HR weight
    kappa2: float = 0.08   # RR weight
    kappa3: float = 0.5    # temperature weight
    tauT: float = 22.0    # threshold for treatment

    # --------------------------
    # CLIPPING BOUNDS
    # --------------------------
    # Latent states
    minLatent: float = 0.0
    maxLatent: float = 2.0

    # Observed variables
    minHeartRate: float = 50.0
    maxHeartRate: float = 150.0

    minRespiratoryRate: float = 10.0
    maxRespiratoryRate: float = 40.0

    minTemperature: float = 36.0
    maxTemperature: float = 40.5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clipValue(value: float, minValue: float, maxValue: float) -> float:
    """
    Ensures a value stays within a specified range.

    This is VERY important to prevent:
    - latent variables exploding
    - unrealistic vital signs
    """
    return max(minValue, min(value, maxValue))


# ============================================================
# INITIALIZATION
# ============================================================

def initializePatient(
    severity: float = 0.2,
    infection: float = 0.2,
    respiratoryStress: float = 0.1,
    advocacy: float = 0.0,
) -> PatientState:
    """
    Creates a new patient with specified initial latent values.

    No randomness here (deterministic setup).
    """
    return PatientState(
        severity=severity,
        infection=infection,
        respiratoryStress=respiratoryStress,
        advocacy=advocacy,
        treatment=0,
    )


# ============================================================
# OBSERVATION GENERATION
# ============================================================

def generateObservations(
    patientState: PatientState,
    params: SimulationParams,
) -> ObservedState:
    """
    Converts latent state into observed vitals.

    This uses deterministic linear equations (no noise yet).
    """

    # Heart Rate depends on severity and infection
    heartRate = (
        params.alpha0
        + params.alpha1 * patientState.severity
        + params.alpha2 * patientState.infection
    )

    # Respiratory Rate depends on severity and respiratory stress
    respiratoryRate = (
        params.beta0
        + params.beta1 * patientState.severity
        + params.beta2 * patientState.respiratoryStress
    )

    # Temperature depends on infection
    temperature = params.delta0 + params.delta1 * patientState.infection

    # Clip values to realistic ranges
    heartRate = clipValue(heartRate, params.minHeartRate, params.maxHeartRate)
    respiratoryRate = clipValue(
        respiratoryRate, params.minRespiratoryRate, params.maxRespiratoryRate
    )
    temperature = clipValue(temperature, params.minTemperature, params.maxTemperature)

    return ObservedState(
        heartRate=heartRate,
        respiratoryRate=respiratoryRate,
        temperature=temperature,
    )


# ============================================================
# TREATMENT LOGIC
# ============================================================

def computeConcernScore(
    observedState: ObservedState,
    params: SimulationParams,
) -> float:
    """
    Computes a simple clinical concern score from observed vitals.

    This acts as a proxy for clinician decision-making.
    """
    return (
        params.kappa1 * observedState.heartRate
        + params.kappa2 * observedState.respiratoryRate
        + params.kappa3 * observedState.temperature
    )


def updateTreatment(
    patientState: PatientState,
    observedState: ObservedState,
    params: SimulationParams,
) -> int:
    """
    Determines whether treatment should be activated.

    Once treatment is triggered, it stays on.
    """
    concernScore = computeConcernScore(observedState, params)

    # If above threshold → start treatment
    if concernScore > params.tauT:
        return 1

    # Otherwise keep previous treatment state
    return patientState.treatment


# ============================================================
# LATENT STATE UPDATE
# ============================================================

def updateLatentState(
    patientState: PatientState,
    params: SimulationParams,
) -> PatientState:
    """
    Evolves the latent state forward one time step.

    Uses deterministic equations (no noise yet).
    """

    # Severity evolves from infection + respiratory stress - treatment
    nextSeverity = (
        patientState.severity
        + params.a1 * patientState.infection
        + params.a2 * patientState.respiratoryStress
        - params.a3 * patientState.treatment
    )

    # Infection grows unless treated
    nextInfection = (
        patientState.infection
        + params.b1
        - params.b2 * patientState.treatment
    )

    # Respiratory stress depends on severity and infection
    nextRespiratoryStress = (
        patientState.respiratoryStress
        + params.c1 * patientState.severity
        + params.c2 * patientState.infection
        - params.c3 * patientState.treatment
    )

    # Clip latent values to safe ranges
    nextSeverity = clipValue(nextSeverity, params.minLatent, params.maxLatent)
    nextInfection = clipValue(nextInfection, params.minLatent, params.maxLatent)
    nextRespiratoryStress = clipValue(
        nextRespiratoryStress, params.minLatent, params.maxLatent
    )

    return PatientState(
        severity=nextSeverity,
        infection=nextInfection,
        respiratoryStress=nextRespiratoryStress,
        advocacy=patientState.advocacy,
        treatment=patientState.treatment,
    )


# ============================================================
# MAIN SIMULATION LOOP (DETERMINISTIC)
# ============================================================

def simulateSinglePatientDeterministic(
    initialState: PatientState,
    params: SimulationParams,
    numSteps: int = 5,
) -> List[dict]:
    """
    Runs a deterministic simulation for a single patient.

    This:
    - generates observations
    - updates treatment
    - updates latent state
    - repeats for numSteps

    Returns a list of dictionaries for easy inspection.
    """

    history: List[dict] = []
    currentState = initialState

    for stepIdx in range(numSteps):

        # Generate observed vitals from latent state
        observedState = generateObservations(currentState, params)

        # Update treatment based on observations
        updatedTreatment = updateTreatment(currentState, observedState, params)
        currentState.treatment = updatedTreatment

        # Store everything for debugging / inspection
        history.append(
            {
                "step": stepIdx,
                "severity": currentState.severity,
                "infection": currentState.infection,
                "respiratoryStress": currentState.respiratoryStress,
                "advocacy": currentState.advocacy,
                "treatment": currentState.treatment,
                "heartRate": observedState.heartRate,
                "respiratoryRate": observedState.respiratoryRate,
                "temperature": observedState.temperature,
                "concernScore": computeConcernScore(observedState, params),
            }
        )

        # Move to next latent state
        currentState = updateLatentState(currentState, params)

    return history