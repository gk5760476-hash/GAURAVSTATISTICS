import sys
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def format_latex(formula):
    # Order matters: replace specific patterns first
    
    # Clean up standard cases first
    formula = formula.replace('\\begin{cases}', '{ ').replace('\\end{cases}', '')
    formula = formula.replace('& \\text{if }', ' if ').replace('\\\\', '; ')
    
    # 1. Fractions
    # \frac{a}{b} -> (a / b)
    while '\\frac' in formula:
        match = re.search(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', formula)
        if match:
            num, denom = match.group(1), match.group(2)
            formula = formula.replace(match.group(0), f"({num} / {denom})")
        else:
            break
            
    # 2. Sums and integrals
    formula = formula.replace('\\int_{-\\infty}^{x}', '∫ [from -∞ to x]')
    formula = formula.replace('\\int_{-\\infty}^{\\frac{x - \\mu}{\\sigma}}', '∫ [from -∞ to (x - μ)/σ]')
    formula = formula.replace('\\int', '∫')
    formula = formula.replace('\\sum_{i=1}^{n}', 'Σ [from i=1 to n]')
    formula = formula.replace('\\sum', 'Σ')
    formula = formula.replace('\\sup_{x \\in \\mathbb{R}}', 'sup [x ∈ ℝ]')
    formula = formula.replace('\\sup_{x}', 'sup [x]')
    formula = formula.replace('\\sup', 'sup')
    
    # 3. Limits
    formula = formula.replace('\\lim_{n \\to \\infty}', 'lim (n → ∞)')
    
    # 4. Braces/delimiters
    formula = formula.replace('\\left(', '(').replace('\\right)', ')')
    formula = formula.replace('\\left[', '[').replace('\\right]', ']')
    formula = formula.replace('\\left\\{', '{').replace('\\right\\}', '}')
    formula = formula.replace('\\right.', '')
    
    # 5. Greek letters and math symbols
    replacements = {
        '\\alpha': 'α',
        '\\beta': 'β',
        '\\gamma': 'γ',
        '\\Gamma': 'Γ',
        '\\delta': 'δ',
        '\\epsilon': 'ε',
        '\\theta': 'θ',
        '\\mu': 'μ',
        '\\sigma': 'σ',
        '\\nu': 'ν',
        '\\pi': 'π',
        '\\Phi': 'Φ',
        '\\le': '≤',
        '\\leq': '≤',
        '\\ge': '≥',
        '\\geq': '≥',
        '\\infty': '∞',
        '\\approx': '≈',
        '\\to': '→',
        '\\implies': '⇒',
        '\\Pr': 'Pr',
        '\\mathbb{R}': 'ℝ',
        '\\mathcal{N}': 'N',
        '\\hat{\\mu}': 'μ-hat',
        '\\hat{\\sigma}': 'σ-hat',
        '\\hat{\\nu}': 'ν-hat',
        '\\hat{s}': 's-hat',
        '\\text{VaR}': 'VaR',
        '\\text{Normal}': 'Normal',
        '\\text{Student-t}': 'Student-t',
        '\\text{model}': 'model',
        '\\text{fitted}': 'fitted',
        '\\text{a.s.}': 'a.s.',
        '\\dots': '...',
        '\\cdot': '·',
        '\\,': ' ',
        '\\;': ' ',
        '\\quad': '  ',
        '\\{': '{',
        '\\}': '}',
        '\\in': '∈',
        '\\ne': '≠',
        '\\neq': '≠',
        '\\times': '×',
        '\\sqrt{\\pi \\nu}': '√(π * ν)',
        '\\sqrt{2\\pi}': '√(2π)',
        '\\sqrt': '√',
    }
    
    for key, value in replacements.items():
        formula = formula.replace(key, value)
        
    # Clean up any leftover LaTeX commands
    formula = re.sub(r'\\([a-zA-Z]+)', r'\1', formula)
    # Format subscripts/superscripts to be cleaner in plain text
    formula = re.sub(r'_(?:\\)?\{([^{}]+)\}', r'_\1', formula)
    formula = re.sub(r'\^(?:\\)?\{([^{}]+)\}', r'^\1', formula)
    
    return formula.strip()

def clean_markdown_text(text):
    # Inline math detection: find $...$ and replace with formatted text in italics
    def replace_inline_math(match):
        formula = match.group(1)
        return f"<i>{format_latex(formula)}</i>"
    
    text = re.sub(r'\$([^\$]+)\$', replace_inline_math, text)

    # Replace bold indicators
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace italic indicators
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Replace code/ticker blocks
    text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#1e293b"><b>\1</b></font>', text)
    return text.strip()

def build_pdf(md_filepath, pdf_filepath):
    # Set page margins
    doc = SimpleDocTemplate(
        pdf_filepath,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1, # Centered
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1f2937'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#4b5563'),
        alignment=1, # Centered
        spaceAfter=15
    )

    math_display_style = ParagraphStyle(
        'MathDisplay',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=1, # Centered
        spaceBefore=8,
        spaceAfter=8,
        leftIndent=20,
        rightIndent=20
    )

    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1f2937')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []

    with open(md_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_table = False
    table_data = []
    
    # Simple markdown parser
    for line in lines:
        stripped = line.strip()
        
        # Table detection
        if stripped.startswith('|'):
            if '---' in stripped:
                continue
            cells = [clean_markdown_text(c.strip()) for c in stripped.split('|')[1:-1]]
            if not in_table:
                in_table = True
                table_data = [cells]
            else:
                table_data.append(cells)
            continue
        else:
            if in_table:
                formatted_table_data = []
                for row_idx, row in enumerate(table_data):
                    formatted_row = []
                    for col in row:
                        if row_idx == 0:
                            formatted_row.append(Paragraph(col, table_header_style))
                        else:
                            formatted_row.append(Paragraph(col, table_text_style))
                    formatted_table_data.append(formatted_row)
                
                t = Table(formatted_table_data, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ('TOPPADDING', (0,0), (-1,-1), 5),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')])
                ]))
                story.append(t)
                story.append(Spacer(1, 10))
                in_table = False
                table_data = []

        if not stripped:
            continue

        # Display math (Double dollar blocks)
        if stripped.startswith('$$') and stripped.endswith('$$'):
            formula = stripped[2:-2]
            clean_formula = format_latex(formula)
            story.append(Paragraph(clean_formula, math_display_style))
            continue

        # Document Title
        if line.startswith('# '):
            title = clean_markdown_text(line[2:])
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 10))
            continue

        # Header 1 (##)
        if line.startswith('## '):
            header = clean_markdown_text(line[3:])
            story.append(Paragraph(header, h1_style))
            continue

        # Header 2 (###)
        if line.startswith('### '):
            header = clean_markdown_text(line[4:])
            story.append(Paragraph(header, h2_style))
            continue

        # Meta info (Course/Assessment)
        if line.startswith('**Course:**') or line.startswith('**Assessment:**'):
            meta = clean_markdown_text(line)
            story.append(Paragraph(meta, meta_style))
            continue

        # Horizontal separator
        if stripped == '---':
            story.append(Spacer(1, 5))
            continue

        # Image check
        img_match = re.match(r'!\[.*?\]\((.*?)\)', stripped)
        if img_match:
            img_path = img_match.group(1).replace('./', '')
            try:
                img = Image(img_path, width=5.5*inch, height=3.4*inch)
                img.hAlign = 'CENTER'
                story.append(Spacer(1, 10))
                story.append(img)
                story.append(Spacer(1, 10))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
            continue

        # Bullets / lists
        if stripped.startswith('* ') or stripped.startswith('- '):
            bullet_text = clean_markdown_text(stripped[2:])
            story.append(Paragraph(f"&bull; {bullet_text}", bullet_style))
            continue
        elif re.match(r'^\d+\.\s', stripped):
            num_match = re.match(r'^(\d+)\.\s(.*)', stripped)
            num = num_match.group(1)
            bullet_text = clean_markdown_text(num_match.group(2))
            story.append(Paragraph(f"{num}. {bullet_text}", bullet_style))
            continue

        # Standard Paragraph
        p_text = clean_markdown_text(stripped)
        story.append(Paragraph(p_text, body_style))

    # Page number generator callback
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8.5)
        canvas.setFillColor(colors.HexColor('#6b7280'))
        page_num = canvas.getPageNumber()
        canvas.drawRightString(8.5 * inch - 0.75 * inch, 0.4 * inch, f"Page {page_num}")
        canvas.drawString(0.75 * inch, 0.4 * inch, "Quantitative Financial Risk Engine Report")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Successfully generated PDF at {pdf_filepath}")

if __name__ == '__main__':
    build_pdf('report.md', 'report.pdf')
