# مسار — Masari Intelligence Core
## منصة الخدمات اللوجستية والمساعدة القانونية الفلسطينية المدعومة بالذكاء الاصطناعي
### A Palestinian AI Platform for Logistics, Mobility Intelligence & Legal Aid

---

> *"المعلومة في وقتها حق، وحقك أن تعرفه."*
> *"Timely information is a right — and knowing your rights is a right."*

---

## Abstract / الملخص

**مسار (Masari)** is an open-source AI platform designed for Palestinians navigating the
complex realities of movement restrictions, checkpoint networks, and property rights under
occupation. Built as a Telegram-first system with a decoupled FastAPI backend, Masari
deploys three specialized AI agents that understand Palestinian dialect, Hebrew loanwords,
and English code-switching — the natural linguistic reality of everyday Palestinian life.

---

## 1. Problem Statement / المشكلة

Palestinians in the West Bank and Gaza face a unique convergence of challenges:

| Domain | Challenge |
|--------|-----------|
| **Mobility** | 593+ obstacles including checkpoints, flying checkpoints, road closures (UN OCHA 2024) |
| **Logistics** | No real-time, Arabic-native platform for cargo/passenger transport coordination |
| **Legal** | Complex overlay of military orders, Jordanian law, PA law, and international law |
| **Linguistic** | Unique code-switching: Palestinian Arabic + Hebrew security terminology + English bureaucracy |

Existing solutions (Google Maps, WhatsApp groups, paper permits) are fragmented, inaccessible,
and unable to handle the specialized vocabulary of occupation — terms like مخصوم (checkpoint),
طيار (flying checkpoint), تصريح (permit), and سيدة (road closure) are absent from standard NLP models.

---

## 2. System Architecture / هندسة النظام

```
┌─────────────────────────────────────────────────────┐
│                  MASARI PLATFORM                      │
├──────────────────────┬──────────────────────────────┤
│   TELEGRAM BOT       │   FLUTTER APP (Future)        │
│   (Primary Interface)│   (Secondary Interface)       │
└──────────┬───────────┴────────────────┬─────────────┘
           │                            │
           ▼                            ▼
┌──────────────────────────────────────────────────────┐
│              FastAPI Gateway (REST API)               │
│         /api/v1/agent  •  /api/v1/detect-and-route   │
│         /webhook/telegram  •  /api/v1/rag/index      │
└──────────────────┬───────────────────────────────────┘
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌──────────┐
  │  ناقل   │ │ الحارس  │ │ المستشار │
  │Transport│ │Watchman │ │Consultant│
  │ Agent   │ │ Agent   │ │  Agent   │
  └────┬────┘ └────┬────┘ └────┬─────┘
       │           │           │
       ▼           ▼           ▼
  ┌──────────────────────────────────┐
  │   GPT-4o / LLM Core             │
  │   Palestinian Dialect Prompts   │
  │   Pydantic Schema Validation    │
  │   Self-Correction Loop          │
  └──────────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   FAKHM RAG PIPELINE   │
                  │  ┌──────────────────┐  │
                  │  │ Parent Doc Store │  │
                  │  │ (Chroma/Redis)   │  │
                  │  ├──────────────────┤  │
                  │  │ BM25 (60%)       │  │
                  │  │ + Dense (40%)    │  │
                  │  ├──────────────────┤  │
                  │  │ BAAI Reranker    │  │
                  │  │ (bge-v2-m3)      │  │
                  │  └──────────────────┘  │
                  └────────────────────────┘
```

---

## 3. The Three Agents / الوكلاء الثلاثة

### 3.1 ناقل — The Transporter
**Mission:** Real-time transport coordination for passengers and cargo across Palestinian territories.

**Inputs:** Free-form Palestinian dialect messages describing transport needs.

**Outputs:** Structured `TransportRequest` with:
- Disambiguated origin/destination (handles 50+ ambiguous place names)
- Cargo classification (passengers / agricultural / medical / commercial)
- Permit requirement detection (tasreeh flag)
- Confidence-weighted extraction with clarification loops

**Novel Contribution:** First Arabic NLP system trained to handle the full register of
Palestinian transport vocabulary including Hebrew road codes (Route 60, Route 443, DCO)
and permit system terminology.

---

### 3.2 الحارس — The Watchman
**Mission:** Crowdsourced, structured checkpoint and road closure intelligence.

**Inputs:** Citizen reports via Telegram (text, voice-to-text).

**Outputs:** Structured `CheckpointReport` with:
- Checkpoint taxonomy (permanent / flying / closure / earth mound)
- Severity assessment (open → closed scale)
- Delay estimates and alternative routes
- Reporter anonymization (SHA-256 hash)
- Cross-verification tracking (verified flag at ≥2 reports)

**Social Impact:** Replaces fragmented WhatsApp groups with a structured, searchable
intelligence layer that can power route planning and alert systems.

---

### 3.3 المستشار — The Consultant
**Mission:** Accessible legal guidance on Housing, Land & Property (HLP) rights,
movement restrictions, detention rights, and permit systems.

**Architecture:** Fakhm (فاهم) RAG Pipeline:
1. **Parent Document Retrieval** — preserves full article context (Geneva Convention, Military Orders)
2. **Hybrid Search** — BM25 (60%) for legal article numbers + Dense vectors (40%) for semantic matching
3. **BAAI/bge-reranker-v2-m3** — cross-encoder reranking of top 20 → best 5 chunks
4. **Multi-query expansion** — Arabic + English variants for cross-lingual legal retrieval
5. **Anti-hallucination guard** — confidence < 0.5 triggers explicit disclaimer + referral

**Legal Corpus:**
- Geneva Convention IV (1949) — English + Arabic
- Hague Regulations (1907)
- UN Basic Principles on Housing & Property Restitution (Pinheiro Principles)
- Palestinian Basic Law (2003/2005)
- Selected Israeli Military Orders (418, 947)
- UNRWA publications
- Al-Haq legal reports

---

## 4. Palestinian NLP Methodology / المنهجية اللغوية

### 4.1 Dialect Handling
Palestinian Arabic (PA) is a Southern Levantine dialect with unique features:

| Feature | Example | Handling |
|---------|---------|----------|
| Definite article elision | "البيت" → "البيت" | Standard |
| Hebrew loan nouns (security) | مخصوم، شباك، تيئور | Custom lexicon in system prompt |
| Hebrew loan verbs | بيتأرر (to coordinate) | Dialect verb normalization |
| English bureaucratic terms | permit, DCO, ID | Code-switch token classification |
| Number system mixing | ٣ أشخاص / 3 persons | Unicode normalization |

### 4.2 Location Disambiguation
A custom disambiguation table covers 30+ ambiguous Palestinian place names:

```python
AMBIGUOUS_LOCATIONS = {
    "المنارة": ["رام الله", "نابلس", "القدس"],
    "البلد": ["أي مدينة"],
    "الجامعة": ["بيرزيت", "النجاح", "الخليل", "القدس"],
    "الحاجز": ["dozens of checkpoints — require road/village context"],
    # ... 26 more entries
}
```

When ambiguity is detected, the system generates a targeted clarification question in dialect:
> *"عفواً، قصدك دوار المنارة في رام الله ولا نابلس؟"*

### 4.3 Self-Correction Loop
```
┌──────────────────────────────────────────┐
│  User Message                            │
│         │                                │
│         ▼                                │
│  Initial Extraction (LLM Pass 1)        │
│         │                                │
│         ▼                                │
│  Confidence Check for each field        │
│         │                                │
│    ┌────┴────────────────────┐           │
│    │ conf < 0.70?            │           │
│    ▼                         ▼           │
│  Generate              Return            │
│  Clarification         Response          │
│  Question (Pass 2)     (status: ok)      │
│    │                                     │
│    ▼                                     │
│  Return Response                         │
│  (status: needs_clarification)           │
└──────────────────────────────────────────┘
```

---

## 5. Evaluation Framework / إطار التقييم

### 5.1 Metrics
| Metric | Formula | Threshold |
|--------|---------|-----------|
| **Faithfulness** (Legal) | Grounded sentences / Total sentences | ≥ 0.80 |
| **Answer Relevance** | Proxy via confidence score | ≥ 0.75 |
| **Field Coverage** | Present fields / Expected fields | ≥ 0.85 |
| **Clarification Accuracy** | Correct ambiguity detection rate | ≥ 0.70 |
| **Latency P95** | 95th percentile response time | ≤ 8 seconds |

### 5.2 Synthetic Test Dataset
50 cases generated across three distribution sets:
- **Transport (40%):** 20 cases covering dialect variations, code-switching, ambiguous locations
- **Checkpoint (30%):** 15 cases covering all checkpoint types and severity levels  
- **Legal (30%):** 15 cases covering HLP, permits, detention, displacement

### 5.3 RAGAS Implementation
The evaluation script (`evaluation.py`) implements:
```python
# Faithfulness: Are claims in the answer grounded in retrieved sources?
faithfulness = grounded_sentences / total_sentences

# Answer Relevance: Does the answer address the question?
answer_relevance = confidence_score  # proxy; replace with NLI scorer in v2

# Field Coverage: Are all expected fields populated?
field_coverage = present_fields / expected_fields
```

---

## 6. Technical Stack / المكدس التقني

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **API Framework** | FastAPI + Uvicorn | Async, production-grade |
| **LLM** | GPT-4o (swappable) | Best Arabic understanding |
| **Embeddings** | text-embedding-3-large | Multilingual, high dimension |
| **Vector Store** | Chroma → Qdrant (prod) | Local dev → cloud scale |
| **BM25** | LangChain BM25Retriever | Legal article number recall |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Multilingual, Arabic-capable |
| **Schema Validation** | Pydantic v2 | Type-safe, fast |
| **Bot Interface** | python-telegram-bot | Webhook architecture |
| **Storage** | Redis (docstore) | Persistent parent chunks |

---

## 7. Social Impact / الأثر الاجتماعي

### Immediate Impact
- **Farmers** can coordinate produce transport across checkpoints with real-time permit guidance
- **Patients** receive instant information on medical permit procedures  
- **Families** get structured checkpoint intelligence before travel
- **Lawyers & NGOs** access legal precedents faster via natural-language queries

### Systemic Impact
- **Documentation:** Every checkpoint report creates a timestamped, structured record
  contributing to human rights documentation databases
- **Pattern Analysis:** Aggregate checkpoint data reveals systematic restriction patterns
  (illegal under IHL Art. 27)
- **Legal Empowerment:** Accessible HLP guidance reduces dependence on expensive legal consultation

### Privacy Architecture
- All reporter identities SHA-256 hashed — no linkability
- No conversation storage by default
- Designed for eventual on-device inference (Phi-3 / Gemma-2 Arabic fine-tune roadmap)

---

## 8. Future Roadmap / خارطة الطريق

| Phase | Timeline | Features |
|-------|---------|---------|
| **v1.0** | Q1 2025 | Core three agents, Telegram bot |
| **v1.5** | Q2 2025 | Flutter mobile app, offline mode |
| **v2.0** | Q3 2025 | Arabic fine-tuned LLM (replace GPT-4o), voice input |
| **v2.5** | Q4 2025 | Community verification network, NGO data integration |
| **v3.0** | 2026 | On-device inference, Gaza-specific dialect model |

---

## 9. Ethical Considerations / الاعتبارات الأخلاقية

- **Accuracy over speed:** Clarification loops preferred over hallucinated extraction
- **Explicit uncertainty:** Confidence scores surfaced to users, never hidden
- **Anti-surveillance design:** Reporter anonymization is non-optional
- **Legal disclaimer:** Every legal response includes explicit non-advice disclaimer
- **Community ownership:** Designed for eventual handoff to Palestinian civil society organizations

---

## 10. References / المراجع

1. UN OCHA. (2024). *West Bank Movement and Access Report.*
2. Al-Haq. (2023). *Annexation, Fragmentation, and the Illegality of the Israeli Closure Regime.*
3. BADIL. (2023). *Survey of Palestinian Refugees and Internally Displaced Persons.*
4. Geneva Convention Relative to the Protection of Civilian Persons in Time of War (1949).
5. UN Basic Principles on Housing and Property Restitution (Pinheiro Principles, 2005).
6. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
7. Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
8. Xiao, S. et al. (2023). *C-Pack: Packaged Resources to Advance General Chinese Embedding.* arXiv:2309.07597. (BAAI/bge)

---

*Masari is developed in solidarity with Palestinian communities navigating complexity with dignity.*

*مسار — بُني بالتضامن مع المجتمعات الفلسطينية التي تواجه التعقيد بكرامة.*
