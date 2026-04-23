"""
main.py — Masari Intelligence Core
FastAPI server + Telegram webhook. Decoupled architecture serving
both the Telegram bot and the future Flutter mobile application.
"""

from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import httpx

from agents import MasariAgentRouter
from schemas import AgentRequest, AgentResponse
from rag_engine import get_rag_engine, load_legal_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("masari.main")

TELEGRAM_BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
LEGAL_DOCS_DIR          = os.getenv("LEGAL_DOCS_DIR", "./data/legal_docs")
INDEX_ON_STARTUP        = os.getenv("INDEX_ON_STARTUP", "false").lower() == "true"


# ─────────────────────────────────────────────
#  Intent Detection (keyword-based)
# ─────────────────────────────────────────────

TRANSPORTER_KW = [
    "بدي روح", "بدي اوصل", "محتاج سيارة", "ناقل", "شحنة",
    "بضاعة", "مسافر", "سرفيس", "بوسطة", "رحلة", "وصّلني",
    "بدي اعبّر", "ride", "cargo", "transport", "shipment",
]
WATCHMAN_KW = [
    "حاجز", "مخصوم", "مخسوم", "طيار", "سيدة", "مسكور",
    "إغلاق", "جيب", "دورية", "تأخير", "فتحوا", "سكروا",
    "بوقفوا", "هويات", "checkpoint", "roadblock", "soldiers",
]

def detect_agent(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in TRANSPORTER_KW):
        return "transporter"
    if any(kw in t for kw in WATCHMAN_KW):
        return "watchman"
    return "consultant"


# ─────────────────────────────────────────────
#  Telegram Helpers
# ─────────────────────────────────────────────

async def send_telegram_message(chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={
            "chat_id": chat_id, "text": text, "parse_mode": "Markdown"
        })

def format_response(r: AgentResponse) -> str:
    d = r.data
    if r.status == "error":
        return f"❌ خطأ: {r.message or 'حدث خطأ غير متوقع'}"

    lines = []
    if r.agent == "transporter":
        lines += [
            "🚗 *طلب النقل*",
            f"📍 من: {d.get('origin', {}).get('resolved_name', '—')}",
            f"📍 إلى: {d.get('destination', {}).get('resolved_name', '—')}",
            f"📦 النوع: {d.get('cargo_type', '—')}",
        ]
        if d.get("tasreeh_required"):
            lines.append("⚠️ *تنبيه: هذا الطريق يتطلب تصريح!*")

    elif r.agent == "watchman":
        emoji = {
            "open": "🟢", "slow": "🟡", "partial": "🟠",
            "closed": "🔴", "flying": "🔵"
        }.get(d.get("severity", ""), "⚪")
        lines += [
            f"{emoji} *تقرير حاجز*",
            f"📍 الموقع: {d.get('location', {}).get('resolved_name', '—')}",
            f"🚧 الحالة: {d.get('severity', '—')}",
        ]
        if d.get("estimated_delay_min"):
            lines.append(f"🕐 التأخير: {d['estimated_delay_min']} دقيقة")
        if d.get("alternative_route"):
            lines.append(f"🔄 طريق بديل: {d['alternative_route']}")

    elif r.agent == "consultant":
        lines += [
            "⚖️ *الاستشارة القانونية*",
            d.get("answer_arabic", "—"),
        ]
        sources = d.get("cited_sources", [])
        if sources:
            lines.append(f"\n📚 *المصادر*:")
            for s in sources[:3]:
                ref = s.get("article_ref", "")
                lines.append(f"  • {s.get('document_title','')} {ref}")
        lines.append(f"\n_⚠️ {d.get('disclaimer', '')}_")

    if r.status == "needs_clarification" and r.message:
        lines.append(f"\n❓ *{r.message}*")

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Masari Intelligence Core starting …")
    if INDEX_ON_STARTUP:
        docs = load_legal_documents(LEGAL_DOCS_DIR)
        if docs:
            get_rag_engine().index_documents(docs)
            logger.info(f"✅ Indexed {len(docs)} document pages")
        else:
            logger.warning("⚠️ No legal documents found")
    yield
    logger.info("👋 Masari shutting down")


# ─────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Masari Intelligence Core",
    description="Palestinian logistics & legal aid AI platform",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
router_instance = MasariAgentRouter()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "masari", "version": "1.0.0"}


@app.post("/api/v1/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """Direct agent call — specify agent explicitly."""
    try:
        return await router_instance.route(request)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/detect-and-route")
async def auto_route(body: dict):
    """Auto-detect agent from message content and route."""
    message = body.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="message required")
    req = AgentRequest(
        agent=detect_agent(message),
        message=message,
        user_id=body.get("user_id"),
    )
    return await router_instance.route(req)


@app.post("/api/v1/rag/index")
async def index_docs(background_tasks: BackgroundTasks):
    """Admin: trigger re-indexing of legal documents."""
    def _index():
        docs = load_legal_documents(LEGAL_DOCS_DIR)
        if docs:
            get_rag_engine().index_documents(docs)
    background_tasks.add_task(_index)
    return {"status": "indexing_started"}


# ─────────────────────────────────────────────
#  Telegram Webhook
# ─────────────────────────────────────────────

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    # Verify secret token
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if TELEGRAM_WEBHOOK_SECRET and secret != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    update = await request.json()
    msg = update.get("message", {})
    if not msg:
        return {"ok": True}

    chat_id = msg.get("chat", {}).get("id")
    user_id = str(msg.get("from", {}).get("id", ""))
    text    = msg.get("text", "")

    if not text or not chat_id:
        return {"ok": True}

    if text.strip() == "/start":
        welcome = (
            "مرحباً بك في *مسار* 🇵🇸\n\n"
            "أنا هنا لمساعدتك في:\n"
            "🚗 *نقل الركاب والبضائع* — أخبرني وين بدك تروح\n"
            "🚧 *تقارير الحواجز* — شاركنا أي حاجز أو إغلاق\n"
            "⚖️ *الاستشارة القانونية* — اسأل عن حقوقك في الأرض والسكن\n\n"
            "كيف أقدر أساعدك اليوم؟"
        )
        background_tasks.add_task(send_telegram_message, chat_id, welcome)
        return {"ok": True}

    async def process_and_reply():
        try:
            req = AgentRequest(agent=detect_agent(text), message=text, user_id=user_id)
            resp = await router_instance.route(req)
            await send_telegram_message(chat_id, format_response(resp))
        except Exception as e:
            logger.exception(e)
            await send_telegram_message(chat_id, "❌ حدث خطأ. يرجى المحاولة لاحقاً.")

    background_tasks.add_task(process_and_reply)
    return {"ok": True}


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") == "development",
    )
