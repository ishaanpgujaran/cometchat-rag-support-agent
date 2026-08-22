# app/orders/models.py
# ---------------------
# Pydantic models for the order-lookup tool.
#
# Design rules
# ~~~~~~~~~~~~
# SafeOrderResult is a strict WHITELIST.  Only fields explicitly declared
# here may ever be populated.  Nothing from the raw orders.json record may
# pass through except via one of these fields.
# OrderItem mirrors the customer-safe item sub-fields from the data dictionary:
# name, quantity, final_sale.  The raw sku field is internal and excluded.
# No field should ever carry raw internal text -- the whitelist itself is
# the safety mechanism, not post-hoc stripping.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class OrderItem(BaseModel):
    """Customer-safe representation of a single line item."""

    name: str
    quantity: int
    final_sale: bool


class SafeOrderResult(BaseModel):
    """
    Whitelisted projection of a raw order record.

    IMPORTANT: this model is the security boundary.
    model_dump() on this object can never leak internal fields
    (risk_score, warehouse_note, support_tags, customer PII) because those
    fields are simply not declared here.
    The caller must only project into these fields and must never use
    **raw_record or any dict-merge that could smuggle unlisted keys.
    """

    order_id: str
    found: bool
    membership_tier: Optional[str] = None
    items: Optional[list[OrderItem]] = None
    placed_at: Optional[str] = None
    status: Optional[str] = None
    status_updated_at: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
