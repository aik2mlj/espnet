#!/usr/bin/env python3
import re

# Read the file
with open('view_data.py', 'r') as f:
    content = f.read()

# Remove emojis - this regex matches most common emojis
emoji_pattern = re.compile(r'[📊🔍✅❌🎤👥📈📉➡️🏆⚠️🔢🎯🌍📏📐🤝🔐💾🔄📋]')
content = emoji_pattern.sub('', content)

# Also remove specific emoji combinations that might be missed
emoji_replacements = [
    '✅ ', '❌ ', '📊 ', '🔍 ', '🎤 ', '👥 ', '📈 ', '📉 ', '➡️ ', 
    '🏆 ', '⚠️ ', '🔢 ', '🎯 ', '🌍 ', '📏 ', '📐 ', '🤝 ', '🔐 '
]

for emoji in emoji_replacements:
    content = content.replace(emoji, '')

# Write back to file
with open('view_data.py', 'w') as f:
    f.write(content)

print('Emojis removed from view_data.py') 