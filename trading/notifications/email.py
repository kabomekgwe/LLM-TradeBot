"""Email Notifier - Send alerts via SMTP email.

Sends trading notifications via email using SMTP.
Supports HTML formatting and attachments.
"""

import logging
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from ..config import TradingConfig
from .manager import NotificationLevel


class EmailNotifier:
    """Email notification handler.

    Sends emails using SMTP server.

    Setup (Gmail example):
    1. Enable 2FA on your Google account
    2. Generate App Password (Security → App passwords)
    3. Set config:
       - SMTP_HOST: smtp.gmail.com
       - SMTP_PORT: 587
       - SMTP_USER: your.email@gmail.com
       - SMTP_PASSWORD: app_password
       - EMAIL_TO: recipient@example.com

    Example:
        >>> notifier = EmailNotifier(config)
        >>> await notifier.send("Trade executed", NotificationLevel.SUCCESS)
    """

    def __init__(self, config: TradingConfig):
        """Initialize email notifier.

        Args:
            config: Trading configuration with SMTP settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # SMTP settings
        self.smtp_host = getattr(config, 'smtp_host', '')
        self.smtp_port = getattr(config, 'smtp_port', 587)
        self.smtp_user = getattr(config, 'smtp_user', '')
        self.smtp_password = getattr(config, 'smtp_password', '')
        self.email_to = getattr(config, 'email_to', '')
        self.email_from = getattr(config, 'email_from', self.smtp_user)

        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.email_to]):
            self.logger.warning("Email not fully configured")

    async def send(self, message: str, level: NotificationLevel = NotificationLevel.INFO):
        """Send email notification.

        Args:
            message: Message text (HTML supported)
            level: Notification priority level
        """
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.email_to]):
            self.logger.warning("Email not configured, skipping notification")
            return

        # Create subject based on level
        subject_prefix = {
            NotificationLevel.INFO: "ℹ️ Info",
            NotificationLevel.SUCCESS: "✅ Success",
            NotificationLevel.WARNING: "⚠️ Warning",
            NotificationLevel.ERROR: "❌ Error",
            NotificationLevel.CRITICAL: "🚨 CRITICAL",
        }

        subject = f"{subject_prefix.get(level, 'ℹ️')} - LLM-TradeBot Notification"

        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.email_from
        msg['To'] = self.email_to

        # Add HTML part
        html_content = self._wrap_html(message, level)
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Send email
        try:
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                start_tls=True,
            )

            self.logger.debug(f"Email sent to {self.email_to}")

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

    async def send_daily_report(
        self,
        report_html: str,
        date: str,
    ):
        """Send daily performance report.

        Args:
            report_html: HTML report content
            date: Report date
        """
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.email_to]):
            return

        subject = f"📊 Daily Trading Report - {date}"

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.email_from
        msg['To'] = self.email_to

        html_part = MIMEText(report_html, 'html')
        msg.attach(html_part)

        try:
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                start_tls=True,
            )

            self.logger.info(f"Daily report sent to {self.email_to}")

        except Exception as e:
            self.logger.error(f"Failed to send daily report: {e}")

    def _wrap_html(self, content: str, level: NotificationLevel) -> str:
        """Wrap content in HTML template.

        Args:
            content: Content to wrap
            level: Notification level for styling

        Returns:
            Complete HTML email
        """
        # Map level to color
        colors = {
            NotificationLevel.INFO: "#3498db",  # Blue
            NotificationLevel.SUCCESS: "#2ecc71",  # Green
            NotificationLevel.WARNING: "#f39c12",  # Orange
            NotificationLevel.ERROR: "#e74c3c",  # Red
            NotificationLevel.CRITICAL: "#992d22",  # Dark red
        }

        color = colors.get(level, "#3498db")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .container {{
                    background: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 30px;
                }}
                .header {{
                    background: {color};
                    color: white;
                    padding: 20px;
                    border-radius: 8px 8px 0 0;
                    margin: -30px -30px 20px -30px;
                }}
                .content {{
                    padding: 20px 0;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #e0e0e0;
                    font-size: 12px;
                    color: #888;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #e0e0e0;
                }}
                th {{
                    background: #f5f5f5;
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2 style="margin: 0;">🤖 LLM-TradeBot</h2>
                </div>
                <div class="content">
                    {content}
                </div>
                <div class="footer">
                    <p>This is an automated notification from LLM-TradeBot.</p>
                    <p>Timestamp: {self._get_timestamp()}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
