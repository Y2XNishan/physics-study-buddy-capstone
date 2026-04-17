from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Physics_Study_Buddy_Capstone_Report_Nishan_Kashyap.pdf"
SCREENSHOTS = ROOT / "screenshots"


def register_fonts() -> str:
    arial_path = Path(r"C:\Windows\Fonts\arial.ttf")
    arial_bold_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if arial_path.exists() and arial_bold_path.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(arial_path)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(arial_bold_path)))
        return "Arial"
    return "Helvetica"


def build_styles(base_font: str):
    styles = getSampleStyleSheet()
    heading_font = "Arial-Bold" if base_font == "Arial" else "Helvetica-Bold"

    styles.add(
        ParagraphStyle(
            name="CustomTitle",
            parent=styles["Title"],
            fontName=heading_font,
            fontSize=15,
            leading=18,
            textColor=colors.black,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName=heading_font,
            fontSize=14,
            leading=16,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustified",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=12,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#333333"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletBody",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=12,
            leading=16,
            leftIndent=14,
            bulletIndent=0,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        )
    )
    return styles


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial" if "Arial" in pdfmetrics.getRegisteredFontNames() else "Helvetica", 10)
    canvas.drawRightString(A4[0] - doc.rightMargin, 18, str(doc.page))
    canvas.restoreState()


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    width, height = image.imageWidth, image.imageHeight
    scale = min(max_width / width, max_height / height)
    image.drawWidth = width * scale
    image.drawHeight = height * scale
    return image


def section(title: str, styles) -> list:
    return [Paragraph(title, styles["SectionHeading"])]


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["BulletBody"], bulletText="•")


def build_story(styles):
    story = []

    story.append(
        Paragraph(
            "Physics Study Buddy: A Faithful Agentic AI Assistant for B.Tech Physics Learners",
            styles["CustomTitle"],
        )
    )
    story.append(Paragraph("Student Name: Nishan Kashyap", styles["BodyJustified"]))
    story.append(Paragraph("Roll Number: 23053060", styles["BodyJustified"]))
    story.append(Paragraph("Batch / Program: Agentic AI Course 2026", styles["BodyJustified"]))
    story.append(Spacer(1, 6))

    story += section("Problem Statement", styles)
    story.append(
        Paragraph(
            "B.Tech students often need help revising core physics concepts outside class hours, "
            "especially when preparing for quizzes, assignments, and semester examinations. Generic "
            "chatbots can sound confident while giving incorrect formulas or unsupported explanations, "
            "which makes them risky for academic use. This project solves that problem by building a "
            "grounded Physics Study Buddy that answers only from a curated physics knowledge base, "
            "remembers the current conversation through thread_id, uses tools when needed, and clearly "
            "admits when the answer is outside its scope.",
            styles["BodyJustified"],
        )
    )

    story += section("User and Objective", styles)
    story.append(
        Paragraph(
            "The intended users are B.Tech students who want quick, syllabus-aligned help on "
            "foundational physics topics such as motion, energy, waves, optics, electrostatics, "
            "current electricity, magnetism, thermodynamics, and modern physics. The objective was "
            "to build a complete working capstone project that demonstrates retrieval-augmented "
            "generation, memory, tool usage, evaluation, and deployment in a browser interface.",
            styles["BodyJustified"],
        )
    )

    story += section("Solution and Features", styles)
    story.append(
        Paragraph(
            "The solution is an Agentic AI assistant implemented with LangGraph. The graph contains "
            "eight nodes: memory, router, retrieve, skip, tool, answer, eval, and save. The state "
            "is defined first using a TypedDict, with fields for question, messages, route, retrieved "
            "context, sources, tool_result, answer, faithfulness, eval_retries, and user_name. The "
            "system uses a ChromaDB in-memory collection backed by twelve topic-specific physics "
            "documents, each covering one focused syllabus concept. The assistant retrieves the top "
            "three relevant documents for concept questions, uses tools for arithmetic or date/time "
            "queries, and preserves memory within one conversation by thread_id.",
            styles["BodyJustified"],
        )
    )
    story.append(bullet("LangGraph StateGraph with 8 required nodes", styles))
    story.append(bullet("ChromaDB RAG knowledge base with 12 physics documents", styles))
    story.append(bullet("MemorySaver with thread_id-based conversation continuity", styles))
    story.append(bullet("Calculator and date/time tool routes", styles))
    story.append(bullet("Faithfulness scoring with retry loop", styles))
    story.append(bullet("Streamlit deployment for browser-based use", styles))

    story += section("Tech Stack", styles)
    for item in [
        "Python",
        "LangGraph",
        "ChromaDB",
        "Sentence Transformers (all-MiniLM-L6-v2) with offline TF-IDF fallback",
        "Streamlit",
        "OpenAI or Groq optional integration",
        "Manual baseline evaluation flow aligned with RAGAS-style reporting",
    ]:
        story.append(bullet(item, styles))

    story += section("Screenshot 1: User Interface", styles)
    story.append(fit_image(SCREENSHOTS / "home.png", 6.6 * inch, 3.4 * inch))
    story.append(
        Paragraph(
            "Figure 1: Physics Study Buddy Streamlit interface showing the project title, domain "
            "description, covered topics, and chat input area.",
            styles["Caption"],
        )
    )

    story.append(PageBreak())

    story += section("Architecture Summary", styles)
    architecture_steps = [
        "memory_node stores the latest user turn, applies a sliding window, and extracts the user name if present.",
        "router_node decides whether the question needs retrieval, tool usage, or memory-only handling.",
        "retrieval_node queries ChromaDB and formats the top matching topic chunks.",
        "skip_retrieval_node handles memory-only turns without retrieval.",
        "tool_node returns calculator or date/time output and never raises exceptions.",
        "answer_node generates a grounded answer using only retrieved context or tool output.",
        "eval_node computes faithfulness and triggers retry when needed.",
        "save_node appends the final assistant answer to conversation history.",
    ]
    for step in architecture_steps:
        story.append(bullet(step, styles))

    story += section("Knowledge Base and Topics Covered", styles)
    story.append(
        Paragraph(
            "The knowledge base contains twelve focused documents, each written to answer concrete "
            "physics questions. The covered areas are kinematics, Newton's laws, work-energy, "
            "gravitation, simple harmonic motion, waves and sound, ray optics, electrostatics, "
            "current electricity, magnetism and electromagnetic induction, thermodynamics, and "
            "modern physics with semiconductors.",
            styles["BodyJustified"],
        )
    )

    story += section("Screenshot 2: Retrieval-Based Concept Answer", styles)
    story.append(fit_image(SCREENSHOTS / "doppler.png", 6.6 * inch, 2.9 * inch))
    story.append(
        Paragraph(
            "Figure 2: The assistant answering a conceptual physics query on the Doppler effect using retrieval from the knowledge base.",
            styles["Caption"],
        )
    )
    story.append(fit_image(SCREENSHOTS / "semiconductor.png", 6.6 * inch, 2.9 * inch))
    story.append(
        Paragraph(
            "Figure 3: The assistant answering an electronics-related physics query and showing the retrieved route and faithfulness score.",
            styles["Caption"],
        )
    )

    story.append(PageBreak())

    story += section("Memory and Tool Use", styles)
    story.append(
        Paragraph(
            "One mandatory capability from the helper document was session memory using MemorySaver "
            "and thread_id. This project supports memory-only follow-up questions such as recalling "
            "the user's name in the same conversation. Another mandatory requirement was tool usage "
            "beyond retrieval. The project includes a calculator tool for arithmetic questions and a "
            "date/time tool for current time queries. Tools are implemented safely and return strings "
            "instead of raising exceptions.",
            styles["BodyJustified"],
        )
    )
    story.append(fit_image(SCREENSHOTS / "name_intro.png", 6.6 * inch, 2.1 * inch))
    story.append(
        Paragraph(
            "Figure 4: Memory handling in the assistant where the user introduces their name and the system stores the context within the same thread.",
            styles["Caption"],
        )
    )
    story.append(fit_image(SCREENSHOTS / "name_recall.png", 6.6 * inch, 2.1 * inch))
    story.append(
        Paragraph(
            "Figure 5: Memory-based follow-up response where the assistant correctly recalls the user's name using conversation history.",
            styles["Caption"],
        )
    )
    story.append(fit_image(SCREENSHOTS / "calculator.png", 6.6 * inch, 1.5 * inch))
    story.append(
        Paragraph(
            "Figure 6: Tool-based response demonstrating calculator functionality with route marked as tool.",
            styles["Caption"],
        )
    )

    story += section("Testing and Evaluation", styles)
    story.append(
        Paragraph(
            "The project includes ten domain questions, two red-team questions, and a multi-turn "
            "memory test sequence. Testing checks route correctness, faithfulness score, and whether "
            "the answer stays grounded in the available context. Out-of-scope questions are handled "
            "safely by refusing to guess. A manual baseline evaluation flow is included for five "
            "grounded question-answer pairs, reporting faithfulness, answer relevancy, and context "
            "precision as baseline quality metrics.",
            styles["BodyJustified"],
        )
    )

    story.append(PageBreak())

    story += section("Additional Working Example", styles)
    story.append(fit_image(SCREENSHOTS / "work_energy.png", 6.6 * inch, 2.9 * inch))
    story.append(
        Paragraph(
            "Figure 7: Retrieval-based response for a physics concept question on the work-energy theorem.",
            styles["Caption"],
        )
    )

    story += section("Unique Points", styles)
    story.append(bullet("Follows the helper-document structure exactly: state-first design, isolated nodes, graph routing, memory, tool use, evaluation, and Streamlit deployment.", styles))
    story.append(bullet("Includes an offline-safe fallback so the project still runs even without cloud API keys.", styles))
    story.append(bullet("Uses focused physics documents to reduce vague retrieval behavior and minimize hallucinated formulas.", styles))
    story.append(bullet("Provides both a command-line interface and a browser-based interface.", styles))

    story += section("Future Improvements", styles)
    story.append(
        Paragraph(
            "With more time, I would add topic-wise quiz generation with answer checking and a "
            "formula-aware numerical problem solver. A strong next step would be a unit-aware "
            "reasoning tool that solves structured physics numericals step by step while remaining "
            "grounded in the syllabus knowledge base and explicitly showing the source concepts used.",
            styles["BodyJustified"],
        )
    )

    story += section("Project Files Submitted", styles)
    for item in [
        "agent.py",
        "capstone_streamlit.py",
        "day13_capstone.ipynb",
        "run_capstone_tests.py",
        "physics_study_buddy/",
        "report/project_documentation.md",
    ]:
        story.append(bullet(item, styles))

    return story


def main():
    base_font = register_fonts()
    styles = build_styles(base_font)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=32,
        title="Physics Study Buddy Capstone Report",
        author="Nishan Kashyap",
    )
    story = build_story(styles)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
