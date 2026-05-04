from __future__ import annotations

from rag.section_intent import infer_section_intent


def test_infers_risk_factor_intent_for_supply_chain_risk_query():
    intent = infer_section_intent("What does Apple say about supply chain risks?")

    assert intent.target_sections[0] == "risk_factors"
    assert "supply chain" in intent.matched_terms
    assert intent.confidence > 0.0


def test_infers_market_risk_before_generic_risk_for_currency_query():
    intent = infer_section_intent("How does Apple describe foreign currency exchange risk?")

    assert intent.target_sections[0] == "market_risk"
    assert "foreign currency" in intent.matched_terms


def test_infers_mda_for_liquidity_and_capital_resources_query():
    intent = infer_section_intent(
        "How does Microsoft discuss liquidity, cash, and capital resources?"
    )

    assert intent.target_sections[0] == "md&a"
    assert "liquidity" in intent.matched_terms
