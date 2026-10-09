"""Repair upstream wheel's cp36 metadata to match its py3 filename/ctypes API."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

wheel = Path('/data/hmp/Robotics/TurboVLA_EOVSAM/dependencies/wheels/decord-0.6.0-py3-none-manylinux2010_x86_64.whl')
with zipfile.ZipFile(wheel) as z:
    files = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
meta = 'decord-0.6.0.dist-info/WHEEL'
assert b'Tag: cp36-cp36m-manylinux2010_x86_64' in files[meta]
files[meta] = files[meta].replace(b'Tag: cp36-cp36m-manylinux2010_x86_64', b'Tag: py3-none-manylinux2010_x86_64')
record = 'decord-0.6.0.dist-info/RECORD'
rows = []
for name, data in files.items():
    if name != record:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        rows.append((name, 'sha256=' + digest, str(len(data))))
rows.append((record, '', ''))
text = io.StringIO(newline='')
csv.writer(text).writerows(rows)
files[record] = text.getvalue().encode()
with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, data in files.items():
        z.writestr(name, data)
print(wheel)
