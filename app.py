import google.generativeai as genai
import typing_extensions as typing
import json, time, os, datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Model configuration (can be adjusted within the UI)
model_name = "gemini-2.0-flash-exp"
model_temperature = 0
model_top_p = 0.95
model_max_output_tokens = 8192

# Data structures for book outline
class Subchapter(typing.TypedDict):
    subchapter_title: str

class Chapter(typing.TypedDict):
    chapter_title: str
    subchapters: list[Subchapter]

class BookOutline(typing.TypedDict):
    book_title: str
    chapters: list[Chapter]

# Initialize the model
model = genai.GenerativeModel(model_name)

def create_dynamic_filename(book_title):
    """Generates a dynamic filename based on the book title and timestamp."""
    sanitized_title = "".join(c for c in book_title if c.isalnum() or c in (' ', '.', '-', '_')).rstrip()
    sanitized_title = sanitized_title.replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitized_title}_book_{timestamp}"

def generate_book(book_title, num_chapters, num_subchapters, model_temperature, model_top_p, model_max_output_tokens):
    """Generates the book content and creates PDF and TXT files."""

    # --- Generate Chapter Titles ---
    prompt_chapters = f"""
    Generate {num_chapters} chapter titles and for each chapter title, {num_subchapters} sub-chapter titles for a book about the {book_title}.
    """

    print(f"DEBUG: Generating outline for book {book_title}")

    try:
        result_chapters = model.generate_content(
            prompt_chapters,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=BookOutline,
                temperature=model_temperature,
                top_p=model_top_p,
                max_output_tokens=model_max_output_tokens,
            ),
        )
        chapter_titles = result_chapters.candidates[0].content.parts[0].text
        chapter_titles = eval(chapter_titles)
    except Exception as e:
        return f"Error generating chapter titles: {e}", None, None

    # --- Iteratively Generate Content ---
    book_content_elements = []
    book_content_text = ""

    # --- PDF Setup with ReportLab ---
    pdf_filename = f"{create_dynamic_filename(book_title)}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()

    styles['Title'].fontSize = 24
    styles['Title'].alignment = 1 
    styles['Title'].leading = 30 
    styles['Title'].spaceAfter = inch
    styles['Title'].autoLeading = 'min'

    styles['Heading1'].fontSize = 22
    styles['Heading1'].spaceAfter = inch * 0.5

    styles['Heading2'].fontSize = 16
    styles['Heading2'].spaceAfter = inch * 0.25

    styles.add(ParagraphStyle(name='FirstParagraph',
                              parent=styles['BodyText'],
                              firstLineIndent=0.5 * inch,
                              fontSize=11,
                              leading=14,
                              alignment=4))

    styles['BodyText'].fontSize = 11
    styles['BodyText'].leading = 14
    styles['BodyText'].alignment = 4

    book_content_elements.append(Paragraph(book_title, styles['Title']))
    book_content_text += f"# {book_title}\n\n"

    for chapter_index, chapter in enumerate(chapter_titles["chapters"]):
        chapter_title = chapter["chapter_title"]
        book_content_text += f"## Chapter {chapter_index + 1}: {chapter_title}\n\n"
        book_content_elements.append(Paragraph(f"Chapter {chapter_index + 1}: {chapter_title}", styles['Heading1']))
        print(f"  DEBUG: Starting Chapter {chapter_index + 1}: {chapter_title}")

        for subchapter in chapter["subchapters"]:
            subchapter_title = subchapter["subchapter_title"]
            book_content_elements.append(Paragraph(subchapter_title, styles['Heading2']))
            book_content_text += f"### {subchapter_title}\n\n"
            print(f"    DEBUG: Starting Subchapter {subchapter_title}")

            prompt_sections = f"""
            Write the content for the subchapter titled '{subchapter_title}' in the chapter '{chapter_title}' of a book about {book_title}. Only output the content, don't output the subchapter title, chapter title and book title. Do not use any markdowns. Make it as detailed as you can.
            """

            while True:
                try:
                    result_sections = model.generate_content(
                        prompt_sections,
                        generation_config=genai.GenerationConfig(
                            temperature=model_temperature,
                            top_p=model_top_p,
                            max_output_tokens=model_max_output_tokens,
                        ),
                    )
                    sections_text = result_sections.text
                    sections = sections_text.split('\n\n')
                    for i, section in enumerate(sections):
                        if i == 0:
                            book_content_elements.append(Paragraph(section, styles['FirstParagraph']))
                        else:
                            book_content_elements.append(Paragraph(section, styles['BodyText']))
                        if i < len(sections) - 1:
                            book_content_elements.append(Spacer(1, 6))

                    book_content_text += f"{sections_text.strip()}\n\n"
                    break
                except Exception as e:
                    print(f"    DEBUG: An error occurred during generation: {e}")
                    print("    DEBUG: Retrying in 5 seconds...")
                    time.sleep(5)

    # --- Output the Book ---
    doc.build(book_content_elements)
    txt_filename = f"{create_dynamic_filename(book_title)}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(book_content_text)

    print(f"Book content written to {txt_filename}")
    return "Book generation complete!", pdf_filename, txt_filename

# --- Gradio Interface ---
with gr.Blocks() as interface:
    gr.Markdown("# AI Book Generator")
    with gr.Row():
        with gr.Column():
            book_title_input = gr.Textbox(label="Book Title")
            num_chapters_input = gr.Slider(label="Number of Chapters", minimum=1, maximum=10, step=1, value=1)
            num_subchapters_input = gr.Slider(label="Number of Subchapters per Chapter", minimum=1, maximum=10, step=1, value=5)
            model_temperature_input = gr.Slider(label="Model Temperature", minimum=0, maximum=1, step=0.01, value=0)
            model_top_p_input = gr.Slider(label="Model Top P", minimum=0, maximum=1, step=0.01, value=0.95)
            model_max_output_tokens_input = gr.Slider(label="Model Max Output Tokens", minimum=1024, maximum=8192, step=256, value=8192)
            generate_button = gr.Button("Generate Book")
        with gr.Column():
            output_message = gr.Textbox(label="Status")
            pdf_output = gr.File(label="Generated PDF")
            txt_output = gr.File(label="Generated TXT")

    generate_button.click(
        generate_book,
        inputs=[
            book_title_input,
            num_chapters_input,
            num_subchapters_input,
            model_temperature_input,
            model_top_p_input,
            model_max_output_tokens_input
        ],
        outputs=[output_message, pdf_output, txt_output]
    )

interface.launch()