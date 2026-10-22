import asyncio
import io
import json
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, Settings
from app.database import case_dict, event, initialize, now, transaction
from app.imaging import PREPROCESS_VERSION, ImageValidationError, fingerprint, quality_check, validate_png
from app.predictor import Predictor


class SexInput(BaseModel):
    sex: Literal["M", "F"]
    revision: int = Field(ge=1)


class QualityInput(BaseModel):
    decision: Literal["usable", "reject"]
    note: str = Field(min_length=2, max_length=2000)
    reviewer: str = Field(min_length=1, max_length=80)
    revision: int = Field(ge=1)


class ReviewInput(BaseModel):
    decision: Literal["accepted", "question", "manual"]
    note: str = Field(min_length=2, max_length=4000)
    reviewer: str = Field(min_length=1, max_length=80)
    confirmed_months: float | None = Field(default=None, ge=0, le=600, allow_inf_nan=False)


class AssessmentInput(BaseModel):
    revision: int = Field(ge=1)


def create_app(settings=None):
    settings = settings or Settings.from_env()
    settings.prepare()
    predictor = Predictor(settings.storage / "models")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bone-assessment")

    @asynccontextmanager
    async def lifespan(app):
        initialize(settings.database)
        yield
        executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="Bone Age Research Workbench", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.predictor = predictor

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "PATCH", "PUT", "DELETE"} and origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "不接受跨站修改请求"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def get_case(db, case_id):
        row = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "样本不存在")
        return row

    def evaluation_dict(db, row, revision=None):
        obj = dict(row)
        obj["stale"] = revision is not None and revision != obj["input_revision"]
        obj["heatmap_url"] = (f"/api/evaluations/{obj['id']}/heatmap"
                                if obj["explanation_file"] else None)
        obj.pop("explanation_file", None)
        obj["reviews"] = [dict(r) for r in db.execute(
            "SELECT * FROM reviews WHERE evaluation_id=? ORDER BY created_at DESC,rowid DESC", (obj["id"],))]
        return obj

    @app.get("/api/status")
    def status():
        with transaction(settings.database) as db:
            stats = dict(db.execute("""SELECT count(*) AS total,
                    sum(CASE WHEN qc_status='pending' THEN 1 ELSE 0 END) AS pending,
                    sum(CASE WHEN qc_status='rejected' THEN 1 ELSE 0 END) AS rejected
                    FROM cases""").fetchone())
            stats["assessed"] = db.execute("""SELECT count(DISTINCT c.id) FROM cases c JOIN evaluations e
                ON e.case_id=c.id AND e.input_revision=c.revision
                WHERE e.status IN ('success','partial')""").fetchone()[0]
            stats["reviewed"] = db.execute("""SELECT count(DISTINCT c.id) FROM cases c JOIN evaluations e
                ON e.case_id=c.id AND e.input_revision=c.revision JOIN reviews r ON r.evaluation_id=e.id
                WHERE e.status IN ('success','partial')""").fetchone()[0]
        for key in stats:
            stats[key] = stats[key] or 0
        meta = predictor.metadata()
        audit = None
        audit_path = settings.dataset / "audit.json"
        if audit_path.exists():
            try:
                audit = json.loads(audit_path.read_text())
            except (OSError, ValueError):
                pass
        return {"app_version": "0.1.0", "scope": "single-image-research", "stats": stats,
                "model": meta, "dataset_audit": audit,
                "dataset_archive_available": (settings.dataset / "rsna-bone-age-v2.zip").is_file(),
                "unlabeled_predictions_available": (settings.storage / "unlabeled-predictions.csv").is_file(),
                "privacy": "本机单用户研究环境；正式部署前需配置身份与访问控制"}

    @app.get("/api/cases")
    def list_cases(q: str = "", state: str = "all", limit: int = 100, offset: int = 0):
        if state not in {"all", "pending", "ready", "rejected", "assessed", "reviewed"}:
            raise HTTPException(422, "筛选状态无效")
        limit = max(1, min(limit, 200)); offset = max(0, offset)
        query = "SELECT c.* FROM cases c WHERE 1=1"
        params = []
        if q:
            query += " AND (c.source_image_id LIKE ? OR c.filename LIKE ? OR c.id LIKE ?)"
            params += [f"%{q[:120]}%"] * 3
        if state in {"pending", "ready", "rejected"}:
            query += " AND c.qc_status=?"; params.append(state)
        elif state in {"assessed", "reviewed"}:
            query += " AND EXISTS(SELECT 1 FROM evaluations e WHERE e.case_id=c.id AND e.input_revision=c.revision AND e.status IN ('success','partial')"
            if state == "reviewed":
                query += " AND EXISTS(SELECT 1 FROM reviews r WHERE r.evaluation_id=e.id)"
            query += ")"
        with transaction(settings.database) as db:
            total = db.execute("SELECT count(*) FROM (" + query + ")", params).fetchone()[0]
            rows = db.execute(query + " ORDER BY c.created_at DESC,c.rowid DESC LIMIT ? OFFSET ?",
                              params + [limit, offset]).fetchall()
            items = []
            for row in rows:
                obj = case_dict(row)
                latest = db.execute("SELECT * FROM evaluations WHERE case_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
                                    (row["id"],)).fetchone()
                obj["latest"] = evaluation_dict(db, latest, row["revision"]) if latest else None
                items.append(obj)
        return {"items": items, "total": total, "offset": offset, "limit": limit}

    async def read_upload(file):
        data = bytearray()
        while chunk := await file.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "单张图片不能超过 15 MiB")
        return bytes(data)

    def add_case(data, filename, sex, source_id, sex_source):
        try:
            image = validate_png(data, settings.max_pixels)
        except ImageValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        sha = fingerprint(data)
        flags, metrics = quality_check(image)
        case_id = uuid.uuid4().hex
        safe_name = Path(filename.replace("\\", "/")).name[:160] or "image.png"
        target = settings.storage / "images" / f"{sha}.png"
        with transaction(settings.database) as db:
            duplicate = db.execute("SELECT id FROM cases WHERE image_sha256=?", (sha,)).fetchone()
            if duplicate:
                raise HTTPException(409, {"message": "这张图片已经导入，可打开原记录重新评估", "case_id": duplicate["id"]})
            if not target.exists():
                temporary = target.with_suffix(f".{case_id}.tmp")
                temporary.write_bytes(data)
                os.replace(temporary, target)
            created = now()
            try:
                db.execute("""INSERT INTO cases(id,source_image_id,filename,image_sha256,sex,sex_source,width,height,
                    qc_status,qc_flags,qc_metrics,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (case_id, source_id[:120], safe_name, sha, sex, sex_source, image.width, image.height,
                     "pending", json.dumps(flags, ensure_ascii=False), json.dumps(metrics), created, created))
            except sqlite3.IntegrityError as exc:
                raise HTTPException(409, "同一图片已被另一请求导入，请刷新列表") from exc
            event(db, case_id, "import", {"sha256": sha, "sex": sex, "source": sex_source})
            return case_dict(get_case(db, case_id))

    @app.post("/api/cases", status_code=201)
    async def upload(file: UploadFile = File(...), sex: str = Form(""), source_image_id: str = Form(""),
                     sex_source: Literal["manual", "csv"] = Form("manual")):
        if sex not in {"", "M", "F"}:
            raise HTTPException(422, "性别只能为 M 或 F；未知请暂留空并人工确认")
        data = await read_upload(file)
        return await asyncio.to_thread(add_case, data, file.filename or "image.png", sex or None,
                                       source_image_id, sex_source if sex else "unknown")

    @app.get("/api/cases/{case_id}")
    def case_detail(case_id: str):
        with transaction(settings.database) as db:
            row = get_case(db, case_id)
            obj = case_dict(row)
            obj["evaluations"] = [evaluation_dict(db, r, row["revision"]) for r in db.execute(
                "SELECT * FROM evaluations WHERE case_id=? ORDER BY created_at DESC,rowid DESC", (case_id,))]
            obj["events"] = [{**dict(r), "detail": json.loads(r["detail"])} for r in db.execute(
                "SELECT * FROM events WHERE case_id=? ORDER BY id DESC LIMIT 40", (case_id,))]
            return obj

    @app.get("/api/cases/{case_id}/image")
    def original_image(case_id: str):
        with transaction(settings.database) as db:
            row = get_case(db, case_id)
            path = settings.storage / "images" / f"{row['image_sha256']}.png"
        if not path.is_file():
            raise HTTPException(404, "原始图片文件缺失，请检查存储")
        return FileResponse(path, media_type="image/png")

    @app.patch("/api/cases/{case_id}/sex")
    def update_sex(case_id: str, payload: SexInput):
        with transaction(settings.database) as db:
            row = get_case(db, case_id)
            if row["revision"] != payload.revision:
                raise HTTPException(409, "记录已更新，请刷新后重试")
            if row["sex"] != payload.sex:
                db.execute("UPDATE cases SET sex=?,sex_source='manual',revision=revision+1,updated_at=? WHERE id=?",
                           (payload.sex, now(), case_id))
                event(db, case_id, "sex_changed", {"before": row["sex"], "after": payload.sex})
            return case_dict(get_case(db, case_id))

    @app.post("/api/cases/{case_id}/quality")
    def review_quality(case_id: str, payload: QualityInput):
        if not payload.note.strip() or not payload.reviewer.strip():
            raise HTTPException(422, "请填写复核人和有效说明")
        with transaction(settings.database) as db:
            row = get_case(db, case_id)
            if row["revision"] != payload.revision:
                raise HTTPException(409, "记录已更新，请刷新后重试")
            target = "ready" if payload.decision == "usable" else "rejected"
            db.execute("UPDATE cases SET qc_status=?,qc_note=?,qc_reviewer=?,revision=revision+1,updated_at=? WHERE id=?",
                       (target, payload.note.strip(), payload.reviewer.strip(), now(), case_id))
            event(db, case_id, "quality_review", payload.model_dump())
            return case_dict(get_case(db, case_id))

    def run_assessment(evaluation_id, model_sha):
        try:
            with transaction(settings.database) as db:
                row = db.execute("SELECT * FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
                snapshot = dict(row)
                db.execute("UPDATE evaluations SET status='running' WHERE id=?", (evaluation_id,))
            path = settings.storage / "images" / f"{snapshot['input_sha256']}.png"
            image = validate_png(path.read_bytes(), settings.max_pixels)
            value, heatmap, warning = predictor.assess(image, snapshot["input_sex"], snapshot["model_id"], model_sha)
            heatmap_file = None
            if heatmap is not None:
                heatmap_file = f"{evaluation_id}.png"
                target = settings.storage / "explanations" / heatmap_file
                heatmap.save(target, format="PNG")
            status = "success" if heatmap_file else "partial"
            with transaction(settings.database) as db:
                db.execute("""UPDATE evaluations SET status=?,prediction_months=?,explanation_status=?,
                    explanation_file=?,error=?,finished_at=? WHERE id=?""",
                    (status, value, "available" if heatmap_file else "unavailable", heatmap_file,
                     warning, now(), evaluation_id))
                event(db, snapshot["case_id"], "assessment_finished", {"evaluation_id": evaluation_id, "status": status})
        except Exception as exc:
            # Persist a useful bounded reason, never a traceback or credential-bearing environment.
            message = str(exc)[:400] or "评估失败，请检查模型和原图"
            with transaction(settings.database) as db:
                db.execute("""UPDATE evaluations SET status='failed',explanation_status='unavailable',
                           error=?,finished_at=? WHERE id=?""", (message, now(), evaluation_id))

    @app.post("/api/cases/{case_id}/assess", status_code=202)
    def assess(case_id: str, payload: AssessmentInput):
        meta = predictor.metadata()
        if not meta:
            raise HTTPException(503, "尚无可用的已注册模型；先完成训练和评估，系统不会生成演示预测")
        with transaction(settings.database) as db:
            db.execute("BEGIN IMMEDIATE")
            row = get_case(db, case_id)
            if row["revision"] != payload.revision:
                raise HTTPException(409, "输入已变更，请刷新后重新评估")
            if row["sex"] not in {"M", "F"}:
                raise HTTPException(422, "请先确认性别")
            if row["qc_status"] != "ready":
                raise HTTPException(422, "请先完成图片质量复核并确认可用")
            if db.execute("SELECT 1 FROM evaluations WHERE case_id=? AND status IN ('queued','running')", (case_id,)).fetchone():
                raise HTTPException(409, "该样本已有运行中的任务")
            evaluation_id = uuid.uuid4().hex
            db.execute("""INSERT INTO evaluations(id,case_id,input_revision,input_sha256,input_sex,model_id,
                preprocess_version,status,created_at) VALUES(?,?,?,?,?,?,?,'queued',?)""",
                (evaluation_id, case_id, row["revision"], row["image_sha256"], row["sex"], meta["model_id"],
                 PREPROCESS_VERSION, now()))
            event(db, case_id, "assessment_queued", {"evaluation_id": evaluation_id, "model_id": meta["model_id"]})
        executor.submit(run_assessment, evaluation_id, meta["checkpoint_sha256"])
        return {"id": evaluation_id, "status": "queued"}

    @app.post("/api/evaluations/{evaluation_id}/reviews", status_code=201)
    def review(evaluation_id: str, payload: ReviewInput):
        if not payload.note.strip() or not payload.reviewer.strip():
            raise HTTPException(422, "请填写复核人和有效意见")
        with transaction(settings.database) as db:
            row = db.execute("SELECT * FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "评估不存在")
            case = get_case(db, row["case_id"])
            if row["status"] not in {"success", "partial"}:
                raise HTTPException(422, "该轮没有可复核的有效骨龄结果")
            if row["input_revision"] != case["revision"]:
                raise HTTPException(409, "这是历史输入的结果，请对当前输入重新评估后复核")
            review_id = uuid.uuid4().hex
            db.execute("INSERT INTO reviews VALUES(?,?,?,?,?,?,?)", (review_id, evaluation_id,
                payload.decision, payload.note.strip(), payload.reviewer.strip(), payload.confirmed_months, now()))
            event(db, row["case_id"], "result_review", {"review_id": review_id, **payload.model_dump()})
        return {"id": review_id}

    @app.get("/api/evaluations/{evaluation_id}/heatmap")
    def heatmap(evaluation_id: str):
        with transaction(settings.database) as db:
            row = db.execute("SELECT explanation_file FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
        if row is None or not row["explanation_file"]:
            raise HTTPException(404, "这轮解释尚不可用")
        path = settings.storage / "explanations" / Path(row["explanation_file"]).name
        if not path.is_file():
            raise HTTPException(404, "解释文件缺失")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/dataset-report")
    def dataset_report():
        result = {}
        for key, path in (("audit", settings.dataset / "audit.json"),
                          ("evaluation", settings.storage / "models" / "evaluation.json")):
            try:
                result[key] = json.loads(path.read_text()) if path.is_file() else None
            except (OSError, ValueError):
                result[key] = None
        return result

    @app.get("/api/unlabeled-predictions")
    def export_predictions():
        path = settings.storage / "unlabeled-predictions.csv"
        if not path.is_file():
            raise HTTPException(404, "尚未生成无标签批量预测，请先运行 predict_unlabeled.py")
        return FileResponse(path, media_type="text/csv; charset=utf-8", filename="rsna-unlabeled-predictions.csv")

    @app.get("/api/prd")
    def prd():
        path = ROOT / "docs" / "PRD_V0.2.html"
        if not path.exists():
            raise HTTPException(404, "需求文档未随部署提供")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    app.mount("/assets", StaticFiles(directory=ROOT / "app" / "static"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "app" / "static" / "index.html")

    return app


app = create_app()
