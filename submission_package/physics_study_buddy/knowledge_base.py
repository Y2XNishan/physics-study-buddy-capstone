from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None  # type: ignore[assignment]


PHYSICS_DOCS = [
    {
        "id": "doc_001",
        "topic": "Kinematics and Motion in One Dimension",
        "category": "Mechanics",
        "text": (
            "Kinematics describes motion without asking what causes it. The most common "
            "quantities are displacement, velocity, acceleration, and time. Displacement is "
            "the directed change in position, while distance is the total path length. Average "
            "velocity equals displacement divided by time, but instantaneous velocity refers "
            "to the rate of change at a single moment. Acceleration measures the rate at which "
            "velocity changes. For constant acceleration, four equations are used often: "
            "v = u + at, s = ut + 1/2 at^2, v^2 = u^2 + 2as, and s = ((u + v) / 2) t. Here u "
            "is initial velocity, v is final velocity, s is displacement, a is acceleration, "
            "and t is time. Graphs are important. The slope of a position-time graph gives "
            "velocity, and the slope of a velocity-time graph gives acceleration. The area "
            "under a velocity-time graph gives displacement. These ideas form the base for "
            "more advanced mechanics problems."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Newton's Laws of Motion",
        "text": (
            "Newton's first law says that a body remains at rest or in uniform straight-line "
            "motion unless acted on by an external unbalanced force. This defines inertia. "
            "Newton's second law connects force to acceleration through F = ma for constant "
            "mass systems. The acceleration produced is in the direction of the net force. "
            "Newton's third law states that for every action there is an equal and opposite "
            "reaction. These forces act on different bodies, so they do not cancel each other. "
            "Free body diagrams help identify all the forces on a chosen object, such as weight, "
            "normal reaction, friction, tension, and applied force. Static friction prevents "
            "motion up to a limiting value, while kinetic friction acts during sliding. Solving "
            "mechanics problems often means choosing a system, drawing the forces, resolving "
            "components, and applying the second law separately along each axis."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Work, Energy, and Power",
        "text": (
            "Work is done when a force causes displacement. For a constant force, work equals "
            "W = F s cos(theta), where theta is the angle between force and displacement. "
            "Kinetic energy is the energy of motion and equals 1/2 mv^2. Potential energy is "
            "energy stored due to position or configuration. Near Earth's surface, gravitational "
            "potential energy is U = mgh. The work-energy theorem states that the net work done "
            "on a body equals the change in kinetic energy. Mechanical energy is conserved when "
            "only conservative forces act. In the presence of friction or drag, some mechanical "
            "energy becomes thermal or other forms. Power measures how fast work is done. "
            "Average power equals work divided by time, and instantaneous power can be written "
            "as P = F dot v. Energy ideas are powerful because they often simplify problems that "
            "would otherwise require multiple force equations."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Gravitation",
        "text": (
            "Newton's law of universal gravitation states that every pair of masses attracts "
            "each other with force F = Gm1m2 / r^2. The force acts along the line joining the "
            "centers of the masses. Close to Earth's surface, the acceleration due to gravity is "
            "approximately 9.8 m/s^2. Weight is the gravitational force acting on a body and is "
            "given by W = mg. Gravitational potential energy near Earth is mgh, but for large "
            "distances the more general expression is U = -GMm / r. Orbital motion results when "
            "gravitational attraction provides the centripetal force. Escape velocity is the "
            "minimum speed needed to leave a gravitational field without further propulsion. It "
            "depends on the planet's mass and radius. Gravitation explains falling bodies, "
            "satellite motion, tides, and many astronomical phenomena."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Simple Harmonic Motion and Oscillations",
        "text": (
            "Simple harmonic motion, or SHM, is a periodic motion in which the restoring force "
            "or acceleration is directly proportional to displacement from the mean position and "
            "is always directed toward that mean position. The basic relation is a = -omega^2 x. "
            "Examples include a mass-spring system and small-angle oscillations of a simple "
            "pendulum. Important quantities are amplitude, time period, frequency, and angular "
            "frequency. The time period of a spring-mass system is T = 2pi sqrt(m/k), while the "
            "simple pendulum for small oscillations has T = 2pi sqrt(l/g). In SHM, kinetic and "
            "potential energy continuously convert into each other while total mechanical energy "
            "remains constant if damping is ignored. At the mean position, speed is maximum and "
            "potential energy is minimum. At the extremes, speed is zero and potential energy is "
            "maximum."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Waves and Sound",
        "text": (
            "A wave transfers energy without transferring matter permanently from one place to "
            "another. Mechanical waves require a medium, while electromagnetic waves do not. "
            "Important wave terms include wavelength, frequency, amplitude, velocity, and phase. "
            "The relation between them is v = f lambda. Transverse waves have oscillations "
            "perpendicular to the direction of travel, while longitudinal waves have oscillations "
            "parallel to it. Sound is a longitudinal mechanical wave. Reflection, refraction, "
            "interference, diffraction, and resonance are major wave phenomena. Standing waves "
            "form due to superposition of waves traveling in opposite directions. In air columns "
            "and stretched strings, boundary conditions determine the allowed harmonics. The "
            "Doppler effect refers to the apparent change in frequency due to relative motion "
            "between source and observer. Resonance occurs when a system is driven at its natural "
            "frequency, producing a large amplitude response."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Ray Optics",
        "text": (
            "Ray optics treats light as straight-line rays and explains image formation using "
            "reflection and refraction. The laws of reflection state that the angle of incidence "
            "equals the angle of reflection and that the incident ray, reflected ray, and normal "
            "lie in the same plane. Refraction is governed by Snell's law: n1 sin(theta1) = "
            "n2 sin(theta2). The refractive index tells how much light slows in a medium. "
            "Spherical mirrors and thin lenses form images according to sign conventions and the "
            "mirror or lens formula. For a thin lens, 1/f = 1/v - 1/u, using the usual Cartesian "
            "sign convention. Magnification is m = h_i / h_o = v / u for lenses. Total internal "
            "reflection occurs when light goes from a denser to a rarer medium and the angle of "
            "incidence exceeds the critical angle. Optical instruments such as the microscope "
            "and telescope use combinations of lenses to improve observation."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Electrostatics and Electric Field",
        "text": (
            "Electrostatics deals with charges at rest. Like charges repel and unlike charges "
            "attract. Coulomb's law gives the magnitude of the force between two point charges as "
            "F = k q1 q2 / r^2. The electric field at a point is defined as force per unit "
            "positive test charge, E = F / q. Electric field lines point away from positive "
            "charges and toward negative charges. Electric potential is the work done per unit "
            "charge to bring a test charge from infinity to a point. Equipotential surfaces are "
            "always perpendicular to electric field lines. A dipole consists of two equal and "
            "opposite charges separated by a small distance. Capacitors store charge and energy. "
            "For a parallel-plate capacitor, capacitance depends on geometry and dielectric "
            "properties. The energy stored in a capacitor can be written as 1/2 CV^2."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Current Electricity",
        "text": (
            "Current electricity studies the motion of charges in conductors. Electric current is "
            "the rate of flow of charge, I = Q / t. Potential difference is the work done per "
            "unit charge between two points. Ohm's law states that V = IR for ohmic conductors "
            "when temperature remains constant. Resistance depends on length, cross-sectional "
            "area, and material. Resistivity is a material property. In series circuits, current "
            "is the same through each component and resistances add directly. In parallel "
            "circuits, the voltage across each branch is the same and reciprocal resistances add. "
            "Electrical power is P = VI, and can also be written as I^2 R or V^2 / R. Cells have "
            "emf and internal resistance, so the terminal voltage may differ from the emf when "
            "current flows. Kirchhoff's laws are used to analyze complex circuits involving "
            "multiple loops and junctions."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Magnetism and Electromagnetic Induction",
        "text": (
            "A moving charge or current produces a magnetic field. A current-carrying conductor "
            "placed in a magnetic field may experience a force. The direction is often found with "
            "Fleming's left-hand rule. A charged particle moving in a magnetic field experiences "
            "the Lorentz force, whose magnetic part is q(v x B). Electromagnetic induction occurs "
            "when magnetic flux linked with a circuit changes. Faraday's law states that the "
            "induced emf equals the negative rate of change of magnetic flux. Lenz's law gives "
            "the direction of the induced current and reflects conservation of energy. Self "
            "induction and mutual induction are key ideas in inductors and transformers. Alternating "
            "current changes direction periodically, and transformers use changing magnetic flux "
            "to step voltage up or down efficiently."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Thermodynamics and Heat",
        "text": (
            "Thermodynamics studies heat, work, temperature, and energy transfer. Temperature "
            "measures thermal state, while heat is energy transferred because of a temperature "
            "difference. The first law of thermodynamics is a statement of energy conservation: "
            "Delta Q = Delta U + Delta W, meaning heat supplied becomes change in internal energy "
            "plus work done by the system. Processes such as isothermal, adiabatic, isobaric, "
            "and isochoric are described by how state variables change. The second law explains "
            "why some processes are irreversible and introduces the concept of entropy. Heat "
            "engines convert thermal energy into work, but no engine can be perfectly efficient. "
            "The Carnot engine sets the theoretical upper limit of efficiency between two "
            "temperatures. Thermodynamics connects macroscopic observations with microscopic "
            "particle behavior and is essential in physics, chemistry, and engineering."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Modern Physics and Semiconductors",
        "text": (
            "Modern physics includes quantum ideas, atomic structure, nuclei, and semiconductor "
            "devices. The photoelectric effect showed that light can behave like packets of energy "
            "called photons. Bohr's model explained hydrogen spectral lines using quantized "
            "orbits, though quantum mechanics later provided a deeper explanation. Radioactivity "
            "involves unstable nuclei emitting alpha, beta, or gamma radiation. Nuclear fission "
            "splits a heavy nucleus, while fusion combines light nuclei. Semiconductors have "
            "conductivity between conductors and insulators. Doping creates n-type and p-type "
            "materials by adding suitable impurities. A p-n junction diode allows current mainly "
            "in one direction and is used in rectification. Transistors are semiconductor devices "
            "used for amplification and switching, forming the basis of modern electronics."
        ),
    },
]


class TfidfEmbedder:
    """Fallback embedder used when SentenceTransformer is unavailable."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(list(texts))

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(list(texts))
        return matrix.toarray().astype(float).tolist()


@dataclass
class KnowledgeBase:
    collection: object
    embedder: object
    topics: list[str]
    docs: list[dict]
    embedder_name: str

    def query(self, question: str, top_k: int = 3) -> dict:
        query_embedding = self.embedder.encode([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        formatted_chunks = []
        sources = []
        for doc, metadata, distance in zip(docs, metadatas, distances):
            topic = metadata.get("topic", "Unknown Topic")
            formatted_chunks.append(f"[{topic}]\n{doc}")
            dist_val = float(distance) if distance is not None else 0.0
            if math.isnan(dist_val) or math.isinf(dist_val):
                dist_val = 0.0
            sources.append(
                {
                    "topic": topic,
                    "distance": round(dist_val, 4),
                }
            )
        return {
            "retrieved": "\n\n".join(formatted_chunks),
            "sources": sources,
        }


import logging

logger = logging.getLogger("physics_study_buddy")


def _build_embedder(texts: list[str]) -> tuple[object, str]:
    if SentenceTransformer is not None:
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            embedder.encode(["warmup"], show_progress_bar=False)
            logger.info("Loaded SentenceTransformer embedder successfully")
            return embedder, "SentenceTransformer(all-MiniLM-L6-v2)"
        except Exception as exc:
            logger.debug("SentenceTransformer not loaded: %s", exc)
    logger.info("Using TfidfVectorizer fallback embedder")
    return TfidfEmbedder(texts), "TfidfVectorizer fallback"


def build_knowledge_base() -> KnowledgeBase:
    texts = [doc["text"] for doc in PHYSICS_DOCS]
    embedder, embedder_name = _build_embedder(texts)
    embeddings = embedder.encode(texts)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="physics_study_buddy")
    collection.add(
        ids=[doc["id"] for doc in PHYSICS_DOCS],
        documents=texts,
        metadatas=[{"topic": doc["topic"]} for doc in PHYSICS_DOCS],
        embeddings=embeddings if isinstance(embeddings, list) else np.array(embeddings).tolist(),
    )
    return KnowledgeBase(
        collection=collection,
        embedder=embedder,
        topics=[doc["topic"] for doc in PHYSICS_DOCS],
        docs=PHYSICS_DOCS,
        embedder_name=embedder_name,
    )
