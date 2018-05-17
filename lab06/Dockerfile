FROM jupyter/minimal-notebook

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD start-notebook.sh --NotebookApp.password='sha1:aa252a38c3f4:4ac5ab221fa43d69142ff88bf094078bed52711f'
