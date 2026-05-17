#!/usr/bin/env python3
"""
Apply large notes index display to orchestrator.py
"""
import re

# Read the file
with open(r"c:\Users\Praveen Raj U S\CBU\cosmic-browser-use\orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add new method _format_large_notes_index after _format_notes
new_method = '''
    def _format_large_notes_index(self, browser_state: Optional[Dict[str, Any]]) -> str:
        """Format large notes index for the prompt - shows ALL available large notes."""
        # Access index from browser state if available
        # Note: We'll need to pass this from browser_controller
        # For now, show instruction to use ListLargeNotes
        return """## LARGE NOTES INDEX
Use ListLargeNotes() to see all available large notes.
Use SearchLargeNotes(query) to find specific notes.
Note: Large notes persist even if their pointers are removed from SAVED NOTES due to budget limits."""
'''

# Insert after _format_notes method
pattern = r'(\n    def _format_dialogs\(self, browser_state:)'
content = re.sub(pattern, new_method + r'\1', content)

# Update _build_messages to include large notes index
old_build_messages = r'''(## SAVED NOTES \(Your Knowledge Base\)\n\{self\._format_notes\(context\.get\('browser_state'\)\)\})'''
new_build_messages = r'''\1
{self._format_large_notes_index(context.get('browser_state'))}'''

content = re.sub(old_build_messages, new_build_messages, content)

# Write back
with open(r"c:\Users\Praveen Raj U S\CBU\cosmic-browser-use\orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Applied large notes index display to orchestrator.py")
