"""
agents.py — Masari Intelligence Core
Three specialized agents: Transporter, Watchman, Consultant.
Implements confidence-based self-correction and clarification loops.
"""

from __future__ import annotations
import json
import logging
import hashlib
from typing import Optional
from datetime import datetime

from openai import AsyncOpenAI
from schemas import (
    TransportRequest, CheckpointReport, LegalResponse,
    AgentRequest, AgentResponse, ConfidenceLevel
)
from prompts import (
    TRANSPORTER_SYSTEM_PROMPT,
    WATCHMAN_SYSTEM_PROMPT,
    SELF_CORRECTION_PROMPT,
)
from rag_engine import get_rag_engine

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.70
LLM_MODEL = "gpt-4o"


# ─────────────────────────────────────────────
#  Base Agent
# ─────────────────────────────────────────────

class BaseAgent:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def _llm_call(self, system_prompt: str, user_message: str) -> str:
        """Single structured LLM call. Returns raw string content."""
        response = await self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content or "{}"

    def _parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)

    def _confidence_level(self, score: float) -> ConfidenceLevel:
        if score >= 0.85:
            return ConfidenceLevel.HIGH
        elif score >= CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    async def _self_correct(
        self,
        original_message: str,
        extraction: dict,
        low_confidence_fields: list[str],
    ) -> list[str]:
        """
        Second LLM pass to generate targeted clarification questions
        for any field with confidence < threshold.
        Returns list of Arabic clarification questions.
        """
        prompt = SELF_CORRECTION_PROMPT.format(
            original_message=original_message,
            previous_extraction=json.dumps(extraction, ensure_ascii=False, indent=2),
            low_confidence_fields=", ".join(low_confidence_fields),
        )
        try:
            raw = await self._llm_call(
                "You are a self-correction module. Return ONLY valid JSON.",
                prompt
            )
            data = self._parse_json(raw)
            return [c["clarification_question"] for c in data.get("corrections", [])]
        except Exception as e:
            logger.warning(f"Self-correction failed: {e}")
            return []


# ─────────────────────────────────────────────
#  Agent 1 — The Transporter (ناقل)
# ─────────────────────────────────────────────

class TransporterAgent(BaseAgent):
    """
    Extracts structured transport requests from Palestinian dialect messages.
    Handles ride requests, cargo shipments, permit requirements.
    """

    async def process(self, message: str, user_id: Optional[str] = None) -> AgentResponse:
        # Step 1: Initial extraction
        raw = await self._llm_call(TRANSPORTER_SYSTEM_PROMPT, message)

        try:
            data = self._parse_json(raw)
        except Exception as e:
            logger.error(f"[Transporter] JSON parse error: {e}")
            return AgentResponse(
                agent="transporter",
                status="error",
                data={"error": str(e)},
                message="حدث خطأ في معالجة الطلب. يرجى المحاولة مرة أخرى.",
            )

        data["raw_message"] = message
        data.setdefault("overall_confidence", 0.5)

        # Step 2: Identify low-confidence fields
        low_conf_fields = []
        for loc_field in ["origin", "destination"]:
            loc = data.get(loc_field, {})
            if isinstance(loc, dict) and loc.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
                low_conf_fields.append(loc_field)
                if loc.get("ambiguous"):
                    q = loc.get("clarification_question")
                    if q:
                        data.setdefault("clarification_questions", []).append(q)

        if data.get("overall_confidence", 1.0) < CONFIDENCE_THRESHOLD or low_conf_fields:
            extra_questions = await self._self_correct(message, data, low_conf_fields)
            data.setdefault("clarification_questions", []).extend(extra_questions)

        # Step 3: Validate schema
        try:
            result = TransportRequest(**data)
        except Exception as e:
            logger.error(f"[Transporter] Schema validation error: {e}")
            return AgentResponse(
                agent="transporter",
                status="error",
                data={"raw": data},
                message=f"خطأ في البيانات: {str(e)}",
            )

        status = "needs_clarification" if result.clarification_questions else "ok"

        return AgentResponse(
            agent="transporter",
            status=status,
            data=result.model_dump(),
            message=result.clarification_questions[0] if result.clarification_questions else None,
        )


# ─────────────────────────────────────────────
#  Agent 2 — The Watchman (الحارس)
# ─────────────────────────────────────────────

class WatchmanAgent(BaseAgent):
    """
    Parses and structures checkpoint/roadblock reports.
    Anonymizes reporters and tracks verification status.
    """

    def _anonymize(self, user_id: Optional[str]) -> Optional[str]:
        if not user_id:
            return None
        return hashlib.sha256(f"masari_salt_{user_id}".encode()).hexdigest()[:16]

    async def process(self, message: str, user_id: Optional[str] = None) -> AgentResponse:
        enriched_message = f"[تقرير جديد — {datetime.utcnow().isoformat()}]\n{message}"
        raw = await self._llm_call(WATCHMAN_SYSTEM_PROMPT, enriched_message)

        try:
            data = self._parse_json(raw)
        except Exception as e:
            logger.error(f"[Watchman] JSON parse error: {e}")
            return AgentResponse(
                agent="watchman", status="error",
                data={"error": str(e)}, message="فشل في معالجة التقرير.",
            )

        data["raw_message"] = message
        data["reporter_id"] = self._anonymize(user_id)
        data.setdefault("reported_at", datetime.utcnow().isoformat())
        data.setdefault("overall_confidence", 0.5)

        loc = data.get("location", {})
        low_conf_fields = []
        if isinstance(loc, dict):
            if loc.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
                low_conf_fields.append("location")
            if loc.get("ambiguous"):
                q = loc.get("clarification_question")
                if q:
                    data.setdefault("clarification_questions", []).append(q)

        if low_conf_fields:
            extra_questions = await self._self_correct(message, data, low_conf_fields)
            data.setdefault("clarification_questions", []).extend(extra_questions)

        try:
            result = CheckpointReport(**data)
        except Exception as e:
            logger.error(f"[Watchman] Schema validation error: {e}")
            return AgentResponse(
                agent="watchman", status="error",
                data={"raw": data}, message=f"خطأ: {str(e)}",
            )

        status = "needs_clarification" if result.clarification_questions else "ok"
        return AgentResponse(
            agent="watchman", status=status,
            data=result.model_dump(),
            message=result.clarification_questions[0] if result.clarification_questions else None,
        )


# ─────────────────────────────────────────────
#  Agent 3 — The Consultant (المستشار)
# ─────────────────────────────────────────────

class ConsultantAgent(BaseAgent):
    """Legal advisory agent using the Fakhm RAG pipeline."""

    def __init__(self):
        super().__init__()
        self.rag_engine = get_rag_engine()

    async def process(self, message: str, user_id: Optional[str] = None) -> AgentResponse:
        import asyncio
        loop = asyncio.get_event_loop()
        result: LegalResponse = await loop.run_in_executor(
            None, self.rag_engine.query, message
        )

        status = "ok"
        reply_message = None
        if result.confidence < CONFIDENCE_THRESHOLD:
            status = "needs_clarification"
            reply_message = (
                f"ثقتي في الإجابة منخفضة ({result.confidence:.0%}). "
                "يُرجى التواصل مع منظمة قانونية متخصصة."
            )

        return AgentResponse(
            agent="consultant", status=status,
            data=result.model_dump(), message=reply_message,
        )


# ─────────────────────────────────────────────
#  Agent Router
# ─────────────────────────────────────────────

class MasariAgentRouter:
    """Routes incoming requests to the appropriate specialized agent."""

    def __init__(self):
        self.agents = {
            "transporter": TransporterAgent(),
            "watchman":    WatchmanAgent(),
            "consultant":  ConsultantAgent(),
        }

    async def route(self, request: AgentRequest) -> AgentResponse:
        agent = self.agents.get(request.agent)
        if not agent:
            return AgentResponse(
                agent=request.agent, status="error",
                data={"error": f"Unknown agent: {request.agent}"},
                message="عذراً، هذا الوكيل غير متاح.",
            )
        logger.info(f"Routing to [{request.agent}]: {request.message[:80]}…")
        return await agent.process(request.message, user_id=request.user_id)
