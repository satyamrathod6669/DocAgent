import json
import logging
import shutil
import tempfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from planner import create_plan
from retriever import retrieve
from generator import generate_document_content, generate_shared_context
from doc_builder import build_docx
from chart_generator import build_charts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docagent")

app = FastAPI(title="DocAgent : Autonomous Document Generation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


class DocumentRequest(BaseModel):
    request: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Plain-English description of the document to generate.",
    )


class PipelineError(Exception):
    """Raised when a specific pipeline stage fails, so we can report which stage."""
    def __init__(self, stage: str, original: Exception):
        self.stage = stage
        self.original = original
        super().__init__(f"{stage} failed: {original}")


def _ascii_safe(text: str) -> str:
    """Replace common 'smart' typography characters that break latin-1
    HTTP headers, then strip anything else non-ASCII as a fallback.

    HTTP headers are restricted to latin-1 encoding by spec. LLMs often
    generate 'smart' typography (non-breaking hyphens, curly quotes,
    em-dashes) that isn't valid latin-1, which crashes header encoding
    if passed through raw.
    """
    replacements = {
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii")


@app.post("/generate")
def generate_document(req: DocumentRequest):
    work_dir = tempfile.mkdtemp(prefix="docagent_")
    try:
        # Step 1: Plan
        try:
            plan_data = create_plan(req.request)
        except Exception as e:
            raise PipelineError("planning", e)

        # Step 2: Retrieve
        try:
            doc_type, template = retrieve(plan_data["document_type"])
        except Exception as e:
            raise PipelineError("template retrieval", e)

        # Step 3: Shared context (consistent mock facts across sections + charts)
        shared_context = generate_shared_context(
            req.request, plan_data["document_type"], plan_data["plan"]
        )

        # Step 4: Generate written sections
        try:
            sections = generate_document_content(
                req.request,
                plan_data["document_type"],
                plan_data["plan"],
                template,
                shared_context,
            )
        except Exception as e:
            raise PipelineError("content generation", e)

        # Step 5: Generate charts (budget + timeline), skipped silently on failure
        chart_paths = build_charts(
            req.request, plan_data["document_type"], work_dir, shared_context
        )

        # Step 6: Build the .docx with sections + charts
        try:
            filepath = build_docx(doc_type, sections, chart_paths, output_dir=work_dir)
        except Exception as e:
            raise PipelineError("document assembly", e)

        # Read bytes into memory now, so nothing depends on the file
        # persisting on disk after this request finishes (important on
        # platforms with ephemeral storage, and for concurrent requests).
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        filename = filepath.split("/")[-1]

        return Response(
            content=file_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                # Surface the agent's own plan/reasoning alongside the file,
                # not just the end output.
                "X-Document-Type": doc_type,
                "X-Assumption": _ascii_safe(plan_data.get("assumption", "")),
                "X-Plan-Steps": json.dumps(plan_data["plan"]),
                "Access-Control-Expose-Headers": "X-Document-Type, X-Assumption, X-Plan-Steps, Content-Disposition",
            },
        )

    except PipelineError as e:
        logger.error("Pipeline failure at %s: %s", e.stage, e.original)
        return JSONResponse(
            status_code=502,
            content={
                "error": f"Document generation failed during {e.stage}.",
                "detail": str(e.original),
            },
        )
    except Exception as e:
        logger.exception("Unexpected error in /generate")
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred.", "detail": str(e)},
        )
    finally:
        # Always clean up the temp working directory, whether we succeeded or not.
        shutil.rmtree(work_dir, ignore_errors=True)
