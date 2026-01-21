import streamlit as st
import os
from datetime import datetime
from audio.recorder import start_recording, stop_and_save
from pipeline.meeting_pipeline import process_meeting
from export.email_sender import send_summary_email
from export.pdf_export import save_meeting_pdf
from config.settings import AUDIO_SETTINGS, EMAIL_SETTINGS

# Ensure exports directory exists
os.makedirs('exports', exist_ok=True)

st.set_page_config(page_title="Live Meeting Summarizer", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stButton>button {width: 100%; margin: 5px 0;}
    .recording {color: red; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🎙️ Meeting Summarizer")
    
    if 'recorder' not in st.session_state:
        st.session_state.update({
            'recorder': None,
            'recording': False,
            'transcript': "",
            'summary': "",
            'audio_file': None
        })
    
    with st.sidebar:
        st.header("Export Options")
        
        # PDF Export Section
        if st.session_state.transcript:
            pdf_filename = f"exports/meeting_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            if st.button("💾 Save as PDF"):
                try:
                    if save_meeting_pdf(
                        transcript=st.session_state.transcript,
                        summary=st.session_state.summary,
                        filepath=pdf_filename,
                        title="Meeting Summary"
                    ):
                        st.sidebar.success(f"✓ PDF saved to {pdf_filename}")
                    else:
                        st.sidebar.error("Failed to save PDF")
                except Exception as e:
                    st.sidebar.error(f"Error saving PDF: {str(e)}")
        
        # Email Settings Section
        st.header("Email Settings")
        send_email = st.checkbox("Enable email", value=False)
        
        if send_email:
            if not all([EMAIL_SETTINGS.get('SMTP_SERVER'), EMAIL_SETTINGS.get('SMTP_PORT')]):
                st.warning("⚠️ Configure SMTP settings in .env")
            else:
                st.markdown("""
                **Note:** Use the App Password, NOT your regular Gmail password!
                """)
                
                email_sender = st.text_input("📧 Your Gmail Address", "")
                app_password = st.text_input("🔑 Gmail App Password (16 characters)", type="password", help="Generate App Password from: https://myaccount.google.com/apppasswords")
                email_recipient = st.text_input("📨 Recipient Email", "")
                email_subject = st.text_input("📝 Subject", "Meeting Summary")
                include_pdf = st.checkbox("📎 Attach PDF", value=True)
                
                # Validate App Password format (16 characters)
                app_password_valid = len(app_password.replace(" ", "")) == 16 if app_password else False
                
                if st.session_state.transcript and email_sender and email_recipient and "@" in email_sender and "@" in email_recipient:
                    if not app_password:
                        st.sidebar.error("❌ App Password is required")
                    elif not app_password_valid:
                        st.sidebar.error("❌ Invalid App Password format (should be 16 characters)")
                    elif st.button("✉️ Send Email with App Password"):
                        try:
                            # Create a temporary PDF if requested
                            pdf_attachment = None
                            if include_pdf:
                                pdf_filename = f"exports/temp_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                if save_meeting_pdf(
                                    transcript=st.session_state.transcript,
                                    summary=st.session_state.summary,
                                    filepath=pdf_filename,
                                    title=email_subject
                                ):
                                    pdf_attachment = pdf_filename
                            
                            # Send email with App Password
                            if send_summary_email(
                                sender=email_sender,
                                sender_password=app_password.replace(" ", ""),  # Remove spaces from App Password
                                recipient=email_recipient,
                                subject=email_subject,
                                body=f"Meeting Summary:\n\n{st.session_state.summary}",
                                attachment_path=pdf_attachment
                            ):
                                st.sidebar.success("✅ Email sent successfully using App Password!")
                            else:
                                st.sidebar.error("❌ Failed to send email. Check your App Password and try again.")
                            
                            # Clean up temporary PDF
                            if include_pdf and pdf_attachment and os.path.exists(pdf_attachment):
                                try:
                                    os.remove(pdf_attachment)
                                except Exception as e:
                                    st.sidebar.warning(f"Could not remove temporary PDF: {str(e)}")
                                    
                        except Exception as e:
                            st.sidebar.error(f"❌ Email error: {str(e)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Recording")
        
        if not st.session_state.recording:
            if st.button("🎤 Start", type="primary"):
                st.session_state.recorder = start_recording()
                st.session_state.recording = True
                st.experimental_rerun()
        else:
            if st.button("⏹️ Stop", type="primary"):
                st.session_state.audio_file = stop_and_save(st.session_state.recorder)
                st.session_state.recording = False
                
                with st.spinner("Processing..."):
                    st.session_state.transcript, st.session_state.summary = process_meeting(st.session_state.audio_file)
                st.experimental_rerun()
        
        if st.session_state.recording:
            st.warning("⏺️ Recording...")
    
    with col2:
        st.header("Results")
        
        if st.session_state.transcript:
            st.subheader("Transcript")
            st.text_area("Transcript", st.session_state.transcript, height=200, key="transcript_area")
            
            st.subheader("Summary")
            st.text_area("Summary", st.session_state.summary, height=150, key="summary_area")
            
            if st.button("📋 Copy Transcript"):
                st.session_state.transcript = st.session_state.transcript
                st.experimental_rerun()
                
            if st.button("📋 Copy Summary"):
                st.session_state.summary = st.session_state.summary
                st.experimental_rerun()

if __name__ == "__main__":
    main()