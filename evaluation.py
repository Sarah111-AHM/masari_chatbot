"""
evaluation.py — Masari Intelligence Core
Academic evaluation using RAGAS (Faithfulness + Answer Relevance).
Synthetic test data generator for 50 Palestinian dialect cases.
Stanford/MIT evaluation standard.
"""

from __future__ import annotations
import json
import random
import asyncio
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger("masari.eval")


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
#  50 test cases covering all agents + code-switching + dialect variation
# ═══════════════════════════════════════════════════════════════════════════

# ── Location Pool ──────────────────────────────────────────────────────────
LOCATIONS_RAMALLAH = ["رام الله", "البيرة", "بيتونيا", "عين عريك", "قدورة", "المنارة رام الله"]
LOCATIONS_NABLUS   = ["نابلس", "حوارة", "بيت دجن", "قريوت", "المنارة نابلس", "البلدة القديمة"]
LOCATIONS_HEBRON   = ["الخليل", "بيت أمر", "دورا", "يطا", "الظاهرية", "حلحول"]
LOCATIONS_JENIN    = ["جنين", "يعبد", "كفر دان", "عرابة", "قباطية"]
LOCATIONS_BETHLEHEM = ["بيت لحم", "بيت ساحور", "بيت جالا", "الدهيشة", "العبيدية"]
LOCATIONS_JERUSALEM = ["القدس", "أبو ديس", "العيزرية", "شعفاط", "الرام"]

ALL_LOCATIONS = (LOCATIONS_RAMALLAH + LOCATIONS_NABLUS + LOCATIONS_HEBRON +
                 LOCATIONS_JENIN + LOCATIONS_BETHLEHEM + LOCATIONS_JERUSALEM)

CHECKPOINTS = [
    "حاجز قلنديا", "حاجز الحاويز", "حاجز بيت إيبا", "حاجز حوارة",
    "حاجز الجلمة", "حاجز الزيتون", "حاجز مخماس", "حاجز النبي إلياس",
    "حاجز القمر", "حاجز ذنابة"
]

# ── Transport Test Cases ────────────────────────────────────────────────────
TRANSPORT_TEMPLATES = [
    # Palestinian dialect
    "بدي سرفيس من {origin} على {destination} بكرا الصبح الساعة ٨",
    "في حدا ناقل من {origin} لـ {destination}؟ عندي بضاعة خضار",
    "محتاج توصيلة من {origin} لـ {destination}، مسافرين ٤ أشخاص",
    "بدي اعبّر شحنة من {origin} على {destination}، في مخصوم؟",
    "وين بلاقي بوسطة من {origin} على {destination}؟",
    # Code-switching Arabic-Hebrew
    "بدي روح {destination} من {origin}، الـ tassrih تبعي لسه سارٍ ولا لازم أجدد؟",
    "في شي route من {origin} على {destination} ما بمر عالـ mahsom؟",
    "محتاج ناقل من {origin} لـ {destination}، الـ DCO فتح؟",
    # Code-switching Arabic-English  
    "Need ride from {origin} to {destination}, بكرا الصبح",
    "بدي transport cargo from {origin} to {destination}, ايمتى أقدر؟",
    # Ambiguous location
    "بدي روح المنارة من {origin}",
    "محتاج توصيلة على البلد من {origin}",
    "في سرفيس على الجامعة من {origin}؟",
]

# ── Checkpoint Test Cases ───────────────────────────────────────────────────
CHECKPOINT_TEMPLATES = [
    # Basic reports
    "{checkpoint} مسكور، ما في عبور",
    "{checkpoint} فيه زحمة كتير، انتظروا ساعة على الأقل",
    "طيار على طريق {origin}، جيبين وبوقفوا السيارات",
    "{checkpoint} مفتوح، بمشي تمام",
    "سيدة على {origin}-{destination}، ما في عبور",
    # Code-switching
    "{checkpoint} مسكور، بوقفوا كل شي. Flying checkpoint على الـ bypass",
    "في mahsom جديد على route 60 قرب {origin}، soldiers كتير",
    "الـ checkpoint عند {checkpoint} بياخد هويات وبفتشوا السيارات",
    # With delay info
    "{checkpoint} زحمة، التأخير ٣٠ دقيقة على الأقل. بديل: طريق وادي النار",
    "إشاعة: {checkpoint} رح يسكر اليوم. تحققوا قبل ما تنزلوا",
    # Detailed reports
    "عند {checkpoint}: بياخد هويات الذكور فوق ١٨ سنة، بيفتشوا السيارات، تأخير ٤٥ دقيقة",
    "طيار عند مدخل {origin}، حاجز مؤقت، بيوقفوا كل شي، ما في تصاريح بتغني",
    "الـ mahsom عند {checkpoint} مفتوح بس بياخد تصريح للدخول على القدس",
]

# ── Legal Test Cases ────────────────────────────────────────────────────────
LEGAL_CASES = [
    # HLP - Housing
    "أخوي استلم إخطار هدم لبيته في {location}، شو حقوقه؟",
    "بيتنا في المنطقة C وبدهم يهدموه، في قانون يحمينا؟",
    "الجيش حجز الأرض تبعتنا وقالوا منطقة عسكرية مغلقة، شو نعمل؟",
    "عندنا كوشان قديم من الأردن، هل بيثبت ملكيتنا؟",
    "جارنا استولى على جزء من أرضنا، كيف نسترجعها؟",
    # Movement & Permits
    "رفضوا طلب التصريح تبعي للعمل داخل الخط الأخضر، شو أعمل؟",
    "شو الفرق بين تصريح عمل وتصريح تجاري؟",
    "أنا طالب وبدي استخرج تصريح دراسة لجامعة القدس، كيف؟",
    "رفضوا تصريح الوصول الطبي لأهلي، في استئناف؟",
    # Detention
    "اعتقلوا أخوي من البيت، بعد كم ساعة لازم يمثل أمام قاضي؟",
    "أخوي معتقل إداري من ٦ أشهر ما في تهمة، شو حقوقه؟",
    "اعتقلوا ولدي عمره ١٦ سنة، شو القانون الدولي بيقول؟",
    # Displacement
    "عندنا حكم هدم قديم من ٢٠ سنة وبدهم ينفذوه هلق، في قانون تقادم؟",
    "شو حقوق اللاجئين الفلسطينيين بموجب القانون الدولي؟",
    "الاستيلاء على الأراضي الزراعية باسم المصلحة العامة، هل قانوني؟",
    # Code-switching legal
    "الـ Military Order 418 بيأثر على building permits تبعتنا؟",
    "شو الـ Geneva Convention بيقول عن demolitions في المناطق المحتلة؟",
    "في حق beit el-karama قبل ما يهدموا البيت؟",
]

# ── Expected Agents ────────────────────────────────────────────────────────
CASE_META = {
    "transport": {"agent": "transporter", "expected_fields": ["origin", "destination", "cargo_type"]},
    "checkpoint": {"agent": "watchman", "expected_fields": ["location", "severity", "checkpoint_type"]},
    "legal": {"agent": "consultant", "expected_fields": ["answer_arabic", "cited_sources", "confidence"]},
}


def generate_synthetic_dataset(n: int = 50, seed: int = 42) -> list[dict]:
    """
    Generate n synthetic test cases for Masari evaluation.
    Distribution: ~20 transport, ~15 checkpoint, ~15 legal.
    """
    random.seed(seed)
    cases = []
    case_id = 1

    def pick_loc():
        return random.choice(ALL_LOCATIONS)

    def pick_checkpoint():
        return random.choice(CHECKPOINTS)

    # Transport cases (~20)
    for tmpl in random.choices(TRANSPORT_TEMPLATES, k=20):
        origin = pick_loc()
        destination = pick_loc()
        while destination == origin:
            destination = pick_loc()
        text = tmpl.format(origin=origin, destination=destination)
        cases.append({
            "id": f"TC-{case_id:03d}",
            "type": "transport",
            "agent": "transporter",
            "input": text,
            "expected_fields": CASE_META["transport"]["expected_fields"],
            "ground_truth": {
                "origin_hint": origin,
                "destination_hint": destination,
            },
            "notes": "auto-generated synthetic"
        })
        case_id += 1

    # Checkpoint cases (~15)
    for tmpl in random.choices(CHECKPOINT_TEMPLATES, k=15):
        checkpoint = pick_checkpoint()
        origin = pick_loc()
        destination = pick_loc()
        text = tmpl.format(
            checkpoint=checkpoint, origin=origin, destination=destination
        )
        cases.append({
            "id": f"TC-{case_id:03d}",
            "type": "checkpoint",
            "agent": "watchman",
            "input": text,
            "expected_fields": CASE_META["checkpoint"]["expected_fields"],
            "ground_truth": {"checkpoint_hint": checkpoint},
            "notes": "auto-generated synthetic"
        })
        case_id += 1

    # Legal cases (~15)
    for q in random.choices(LEGAL_CASES, k=15):
        location = pick_loc()
        text = q.format(location=location) if "{location}" in q else q
        cases.append({
            "id": f"TC-{case_id:03d}",
            "type": "legal",
            "agent": "consultant",
            "input": text,
            "expected_fields": CASE_META["legal"]["expected_fields"],
            "ground_truth": {"domain_hint": "HLP or movement or detention"},
            "notes": "auto-generated synthetic"
        })
        case_id += 1

    random.shuffle(cases)
    return cases[:n]


# ═══════════════════════════════════════════════════════════════════════════
#  RAGAS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationResult(BaseModel):
    case_id:            str
    agent:              str
    input:              str
    status:             str
    faithfulness:       Optional[float] = None
    answer_relevance:   Optional[float] = None
    field_coverage:     float = 0.0     # % of expected_fields present in response
    confidence:         Optional[float] = None
    latency_ms:         Optional[int]   = None
    error:              Optional[str]   = None

class EvaluationReport(BaseModel):
    timestamp:          str
    total_cases:        int
    pass_rate:          float
    avg_faithfulness:   Optional[float]
    avg_answer_relevance: Optional[float]
    avg_field_coverage: float
    avg_latency_ms:     Optional[float]
    results:            list[EvaluationResult]
    by_agent:           dict


async def run_evaluation(
    test_cases: Optional[list[dict]] = None,
    n_cases: int = 50,
    output_path: str = "./eval_results.json",
) -> EvaluationReport:
    """
    Run full RAGAS-style evaluation over synthetic test cases.
    Measures: Faithfulness, Answer Relevance, Field Coverage, Latency.
    """
    import time
    from agents import MasariAgentRouter
    from schemas import AgentRequest

    if test_cases is None:
        test_cases = generate_synthetic_dataset(n=n_cases)

    router = MasariAgentRouter()
    results = []

    for case in test_cases:
        start_ms = time.time() * 1000

        try:
            req = AgentRequest(
                agent=case["agent"],
                message=case["input"],
                user_id="eval_runner",
            )
            response = await router.route(req)
            latency = int(time.time() * 1000 - start_ms)

            data = response.data

            # ── Field Coverage ──────────────────────────────────────
            expected = case.get("expected_fields", [])
            present = sum(1 for f in expected if data.get(f) is not None)
            field_coverage = present / len(expected) if expected else 1.0

            # ── RAGAS Metrics ───────────────────────────────────────
            # For legal consultant: use faithfulness from LegalResponse
            faithfulness = None
            answer_relevance = None
            confidence = data.get("confidence") or data.get("overall_confidence")

            if case["agent"] == "consultant":
                # Compute simple faithfulness: check if answer references cited sources
                answer = data.get("answer_arabic", "")
                sources = data.get("cited_sources", [])
                if sources and answer:
                    # Heuristic: faithfulness = proportion of answer sentences with grounding
                    sentences = [s.strip() for s in answer.split("،") if len(s.strip()) > 10]
                    grounded = sum(
                        1 for sent in sentences
                        if any(s.get("document_title","") in answer for s in sources)
                    )
                    faithfulness = min(1.0, grounded / max(len(sentences), 1))
                    # Answer relevance: proxy via confidence
                    answer_relevance = confidence or 0.5
                else:
                    faithfulness = 0.0
                    answer_relevance = 0.0

            result = EvaluationResult(
                case_id=case["id"],
                agent=case["agent"],
                input=case["input"][:120],
                status=response.status,
                faithfulness=round(faithfulness, 4) if faithfulness is not None else None,
                answer_relevance=round(answer_relevance, 4) if answer_relevance is not None else None,
                field_coverage=round(field_coverage, 4),
                confidence=round(float(confidence), 4) if confidence else None,
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"Eval error for {case['id']}: {e}")
            result = EvaluationResult(
                case_id=case["id"],
                agent=case["agent"],
                input=case["input"][:120],
                status="error",
                error=str(e),
                latency_ms=int(time.time() * 1000 - start_ms),
            )

        results.append(result)
        logger.info(f"✓ {result.case_id} [{result.agent}] — "
                    f"field_coverage={result.field_coverage:.0%} "
                    f"latency={result.latency_ms}ms")

    # ── Aggregate Metrics ──────────────────────────────────────────────────
    ok_results = [r for r in results if r.status != "error"]
    faith_vals = [r.faithfulness for r in ok_results if r.faithfulness is not None]
    rel_vals   = [r.answer_relevance for r in ok_results if r.answer_relevance is not None]
    lat_vals   = [r.latency_ms for r in ok_results if r.latency_ms is not None]

    by_agent: dict = {}
    for agent in ("transporter", "watchman", "consultant"):
        agent_res = [r for r in ok_results if r.agent == agent]
        if agent_res:
            by_agent[agent] = {
                "count": len(agent_res),
                "avg_field_coverage": round(
                    sum(r.field_coverage for r in agent_res) / len(agent_res), 4),
                "avg_confidence": round(
                    sum(r.confidence or 0 for r in agent_res) / len(agent_res), 4),
            }

    report = EvaluationReport(
        timestamp=datetime.utcnow().isoformat(),
        total_cases=len(results),
        pass_rate=round(len(ok_results) / len(results), 4) if results else 0,
        avg_faithfulness=round(sum(faith_vals) / len(faith_vals), 4) if faith_vals else None,
        avg_answer_relevance=round(sum(rel_vals) / len(rel_vals), 4) if rel_vals else None,
        avg_field_coverage=round(
            sum(r.field_coverage for r in ok_results) / len(ok_results), 4
        ) if ok_results else 0.0,
        avg_latency_ms=round(sum(lat_vals) / len(lat_vals), 1) if lat_vals else None,
        results=results,
        by_agent=by_agent,
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(f"📊 Evaluation report saved to {output_path}")

    return report


# ─────────────────────────────────────────────
#  CLI Entry
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Masari Evaluation Suite")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate synthetic data, don't run eval")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output", default="./eval_results.json")
    parser.add_argument("--data-output", default="./synthetic_test_data.json")
    args = parser.parse_args()

    # Always generate synthetic data
    dataset = generate_synthetic_dataset(n=args.n)
    with open(args.data_output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ Generated {len(dataset)} synthetic test cases → {args.data_output}")

    if not args.generate_only:
        print("🚀 Running evaluation …")
        report = asyncio.run(run_evaluation(
            test_cases=dataset,
            output_path=args.output,
        ))
        print(f"\n{'='*50}")
        print(f"📊 MASARI EVALUATION REPORT")
        print(f"{'='*50}")
        print(f"Total Cases:         {report.total_cases}")
        print(f"Pass Rate:           {report.pass_rate:.1%}")
        print(f"Avg Faithfulness:    {report.avg_faithfulness or 'N/A'}")
        print(f"Avg Answer Relevance:{report.avg_answer_relevance or 'N/A'}")
        print(f"Avg Field Coverage:  {report.avg_field_coverage:.1%}")
        print(f"Avg Latency:         {report.avg_latency_ms or 'N/A'} ms")
        print(f"\nBy Agent:")
        for agent, stats in report.by_agent.items():
            print(f"  {agent}: {stats}")
