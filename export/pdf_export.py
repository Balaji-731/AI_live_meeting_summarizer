from fpdf import FPDF
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

def save_meeting_pdf(transcript, summary, filepath, title="Meeting Summary"):
    """
    Save meeting transcript and summary to a formatted PDF file.
    
    Args:
        transcript (str): The meeting transcript text
        summary (str): The generated meeting summary
        filepath (str): Path to save the PDF file
        title (str): Title for the PDF document
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create PDF with better formatting
        pdf = FPDF()
        
        # Add a page
        pdf.add_page()
        
        # Set auto page break
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1, 'C')
        
        # Add date
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        
        # Add a line break
        pdf.ln(10)
        
        # Add summary section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Meeting Summary', 0, 1)
        
        pdf.set_font('Arial', '', 12)
        # Split summary into paragraphs and add to PDF
        for para in summary.split('\n'):
            if para.strip():
                pdf.multi_cell(0, 10, para.strip())
                pdf.ln(5)
        
        # Add transcript section
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Full Transcript', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        # Split transcript into lines and add to PDF
        for line in transcript.split('\n'):
            if line.strip():
                pdf.multi_cell(0, 8, line.strip())
                pdf.ln(3)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        # Save the PDF
        pdf.output(filepath)
        logger.info(f"PDF saved successfully: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving PDF to {filepath}: {str(e)}")
        return False
