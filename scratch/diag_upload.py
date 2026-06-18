content = open('dashboard/templates/dashboard/index.html', encoding='utf-8').read()
print('spin keyframe defined:', '@keyframes spin' in content)
print('slideUp defined:', '@keyframes slideUp' in content)
print('augmentDropzone found:', 'augmentDropzone' in content)
print('augmentFileInput found:', 'augmentFileInput' in content)
# Check CSRF exempt on augment_db view
views = open('dashboard/views.py', encoding='utf-8').read()
import re
matches = re.findall(r'@csrf_exempt\s*\ndef\s+(\w+)', views)
print('csrf_exempt views:', matches)
# Check if there's a CSRF token in form or fetch call
print('CSRF in fetch augment:', 'csrf' in content[content.find('augment-db'):content.find('augment-db')+200].lower())
