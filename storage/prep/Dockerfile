FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY *.py ./
COPY prep-jobs.json molecular-inputs.json ./
RUN python -c "import hashlib,pathlib; h=hashlib.sha256(); [(h.update(p.name.encode()+b'\0'),h.update(p.read_bytes())) for p in sorted([*pathlib.Path('.').glob('*.py'),pathlib.Path('prep-jobs.json'),pathlib.Path('molecular-inputs.json')])]; pathlib.Path('build.sha256').write_text(h.hexdigest())"
EXPOSE 8080
CMD ["python", "runner.py"]
