"""
Masari Schemas — مخططات البيانات
Pydantic v2 models for structured extraction and API contracts.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class AgentType(str, Enum):
    TRANSPORTER = "transporter"
    WATCHMAN    = "watchman"
    CONSULTANT  = "consultant"

class CargoType(str, Enum):
    PASSENGER = "passenger"
    GOODS     = "goods"
    MIXED     = "mixed"
    UNKNOWN   = "unknown"

class CheckpointType(str, Enum):
    FIXED   = "fixed"
    FLYING  = "flying"
    CLOSURE = "closure"
    UNKNOWN = "unknown"

class CheckpointStatus(str, Enum):
    OPEN      = "open"
    SLOW      = "slow"
    CONGESTED = "congested"
    CLOSED    = "closed"
    UNKNOWN   = "unknown"

class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # > 0.85
    MEDIUM = "medium"  # 0.70-0.85
    LOW    = "low"     # < 0.70 → triggers clarification


# ═══════════════════════════════════════════════════════════════════════════
#  Shared
# ═══════════════════════════════════════════════════════════════════════════

class Location(BaseModel):
    raw_mention: str
    resolved_name: str
    governorate: Optional[str] = None
    district: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def round_conf(cls, v: float) -> float:
        return round(v, 4)


class FieldConfidence(BaseModel):
    field_name: str
    value: Optional[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Agent 1 — الناقل
# ═══════════════════════════════════════════════════════════════════════════

class TransportRequest(BaseModel):
    raw_text: str
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TransportOffer(BaseModel):
    origin: Location
    destination: Location
    cargo_type: CargoType
    departure_time: Optional[str]  = None
    seats_available: Optional[int] = Field(None, ge=0, le=50)
    weight_kg: Optional[float]     = None
    price_ils: Optional[float]     = None
    contact_info: Optional[str]    = None
    special_notes: Optional[str]   = None
    requires_tasreeh: bool         = False
    permit_zone: Optional[str]     = None
    via_checkpoints: List[str]     = Field(default_factory=list)
    field_confidences: List[FieldConfidence] = Field(default_factory=list)
    overall_confidence: float      = Field(..., ge=0.0, le=1.0)
    raw_input_language: str        = "mixed-ar"


class TransportResponse(BaseModel):
    success: bool = True
    offer: TransportOffer
    formatted_ar: str
    needs_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  Agent 2 — الحارس
# ═══════════════════════════════════════════════════════════════════════════

class CheckpointReport(BaseModel):
    raw_text: str
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CheckpointEvent(BaseModel):
    checkpoint_name: Optional[str]   = None
    checkpoint_type: CheckpointType
    status: CheckpointStatus
    location: Location
    reported_at: datetime            = Field(default_factory=datetime.utcnow)
    delay_minutes: Optional[int]     = Field(None, ge=0)
    military_present: Optional[bool] = None
    special_units: Optional[str]     = None
    alternative_routes: List[str]    = Field(default_factory=list)
    verified_reports: int            = Field(1, ge=1)
    reporter_confidence: float       = Field(..., ge=0.0, le=1.0)
    raw_dialect_terms: List[str]     = Field(default_factory=list)


class CheckpointResponse(BaseModel):
    success: bool = True
    event: CheckpointEvent
    alert_level: Literal["green", "yellow", "red"]
    formatted_ar: str
    broadcast_message: str


# ═══════════════════════════════════════════════════════════════════════════
#  Agent 3 — المستشار
# ═══════════════════════════════════════════════════════════════════════════

class LegalQuery(BaseModel):
    question: str
    user_id: Optional[str] = None
    context_history: List[dict] = Field(default_factory=list)


class LegalSource(BaseModel):
    document_title: str
    article_or_section: Optional[str] = None
    page_number: Optional[int]        = None
    excerpt: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    rerank_score: Optional[float]     = None


class LegalResponse(BaseModel):
    success: bool = True
    answer_ar: str
    answer_en: Optional[str]          = None
    sources: List[LegalSource]
    faithfulness_score: Optional[float]   = None
    answer_relevance: Optional[float]     = None
    confidence_level: ConfidenceLevel
    disclaimer_ar: str = (
        "⚠️ هذه المعلومات للتوعية القانونية فقط ولا تُغني عن استشارة محامٍ مختص."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Self-Correction
# ═══════════════════════════════════════════════════════════════════════════

class ClarificationNeeded(BaseModel):
    success: bool = False
    agent: AgentType
    questions: List[str]
    partial_extraction: Optional[dict]     = None
    confidence_report: List[FieldConfidence] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  Telegram (minimal)
# ═══════════════════════════════════════════════════════════════════════════

class TelegramChat(BaseModel):
    id: int
    type: str

class TelegramMessage(BaseModel):
    message_id: int
    chat: TelegramChat
    text: Optional[str] = None
    date: int

class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[TelegramMessage] = None
