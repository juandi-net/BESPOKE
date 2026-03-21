"""Stage 2c: Trajectory Analysis — read-only queries over the warehouse."""

import json
import sqlite3
from datetime import datetime, timedelta


def _months_filter(months: int | None) -> tuple[str, list]:
    """Return (WHERE clause fragment, params) filtering to the last N months."""
    if months is None:
        return "", []
    cutoff = (datetime.utcnow() - timedelta(days=months * 30)).strftime("%Y-%m-%dT00:00:00Z")
    return "AND captured_at >= ?", [cutoff]


def _domain_filter(domain: str | None) -> tuple[str, list]:
    """Return (WHERE clause fragment, params) filtering to a specific domain."""
    if domain is None:
        return "", []
    return "AND domain = ?", [domain]


def _build_filters(months: int | None = None, domain: str | None = None) -> tuple[str, list]:
    """Combine month and domain filters into a single clause + params."""
    m_clause, m_params = _months_filter(months)
    d_clause, d_params = _domain_filter(domain)
    return f"{m_clause} {d_clause}", m_params + d_params


def domain_distribution(conn: sqlite3.Connection, months: int | None = None) -> dict[str, dict[str, int]]:
    """Monthly interaction counts per domain.

    Returns: {month: {domain: count, ...}, ...}
    """
    f_clause, f_params = _months_filter(months)
    query = f"""
        SELECT
            strftime('%Y-%m', captured_at) as month,
            domain,
            COUNT(*) as count
        FROM interactions
        WHERE domain IS NOT NULL
        {f_clause}
        GROUP BY month, domain
        ORDER BY month, count DESC
    """
    rows = conn.execute(query, f_params).fetchall()
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        month = row["month"]
        if month not in result:
            result[month] = {}
        result[month][row["domain"]] = row["count"]
    return result


def complexity_trajectory(conn: sqlite3.Connection, months: int | None = None, domain: str | None = None) -> list[dict]:
    """Monthly averages of interaction complexity signals.

    Returns: [{month, avg_tokens, avg_primitives, interaction_count}, ...]
    """
    f_clause, f_params = _build_filters(months, domain)
    query = f"""
        SELECT
            strftime('%Y-%m', captured_at) as month,
            AVG(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) as avg_tokens,
            AVG(
                CASE
                    WHEN reasoning_primitives IS NOT NULL AND reasoning_primitives != '[]'
                    THEN json_array_length(reasoning_primitives)
                    ELSE 0
                END
            ) as avg_primitives,
            COUNT(*) as interaction_count
        FROM interactions
        WHERE processed_2a_at IS NOT NULL
        {f_clause}
        GROUP BY month
        ORDER BY month
    """
    return [dict(row) for row in conn.execute(query, f_params).fetchall()]


def standards_evolution(conn: sqlite3.Connection, months: int | None = None, domain: str | None = None) -> list[dict]:
    """Monthly accept/reject rates from feedback classification.

    Returns: [{month, total, accepts, accept_rate}, ...]
    """
    f_clause, f_params = _build_filters(months, domain)
    query = f"""
        SELECT
            strftime('%Y-%m', captured_at) as month,
            feedback_class,
            COUNT(*) as count
        FROM interactions
        WHERE feedback_class IS NOT NULL
        {f_clause}
        GROUP BY month, feedback_class
        ORDER BY month
    """
    rows = conn.execute(query, f_params).fetchall()

    # Group by month
    monthly: dict[str, dict[str, int]] = {}
    for row in rows:
        month = row["month"]
        if month not in monthly:
            monthly[month] = {}
        monthly[month][row["feedback_class"]] = row["count"]

    result = []
    for month in sorted(monthly):
        classes = monthly[month]
        total = sum(classes.values())
        accepts = classes.get("strong_accept", 0) + classes.get("accept", 0)
        result.append({
            "month": month,
            "total": total,
            "accepts": accepts,
            "accept_rate": accepts / total if total > 0 else 0.0,
        })
    return result


def capability_emergence(conn: sqlite3.Connection) -> list[dict]:
    """First occurrence of each domain.

    Returns: [{domain, first_seen, total_interactions}, ...]
    """
    query = """
        SELECT
            domain,
            MIN(captured_at) as first_seen,
            COUNT(*) as total_interactions
        FROM interactions
        WHERE domain IS NOT NULL
        GROUP BY domain
        ORDER BY first_seen
    """
    return [dict(row) for row in conn.execute(query).fetchall()]


def _bar(count: int, max_count: int, max_width: int = 20) -> str:
    """Render a proportional bar using block characters."""
    if max_count == 0:
        return ""
    width = max(1, round(count / max_count * max_width))
    return "\u2588" * width


def _format_number(n: float) -> str:
    """Format a number with comma separators."""
    if n >= 1000:
        return f"{n:,.0f}"
    return f"{n:.1f}" if isinstance(n, float) and n != int(n) else str(int(n))


def generate_highlights() -> None:
    """Print a compact trajectory summary — domain shifts, trends, new domains."""
    from bespoke.db.init import get_connection

    conn = get_connection()

    has_data = conn.execute(
        "SELECT COUNT(*) as n FROM interactions WHERE domain IS NOT NULL"
    ).fetchone()["n"]

    if has_data == 0:
        conn.close()
        return

    lines = []

    # Domain shift summary
    dist = domain_distribution(conn)
    if dist:
        months_sorted = sorted(dist.keys())
        latest = dist[months_sorted[-1]]
        total = sum(latest.values())
        top = sorted(latest.items(), key=lambda x: -x[1])[:4]
        parts = [f"{d} {c * 100 // total}%" for d, c in top]
        line = ", ".join(parts)

        if len(months_sorted) >= 2:
            prev = dist[months_sorted[-2]]
            prev_total = sum(prev.values())
            if prev_total > 0:
                shifts = []
                for d, c in top:
                    curr_pct = c / total * 100
                    prev_pct = prev.get(d, 0) / prev_total * 100
                    delta = curr_pct - prev_pct
                    if abs(delta) >= 5:
                        sign = "+" if delta >= 0 else ""
                        shifts.append(f"{d} {sign}{delta:.0f}pp")
                if shifts:
                    line += f" \u2014 {', '.join(shifts)}"

        lines.append(f"  Domains: {line}")

    # Complexity trend (compare last two months)
    complexity = complexity_trajectory(conn)
    if complexity and len(complexity) >= 2:
        prev_c, curr_c = complexity[-2], complexity[-1]
        trend_parts = []
        arrow = "\u2192"
        if prev_c["avg_tokens"] > 0:
            change = (curr_c["avg_tokens"] - prev_c["avg_tokens"]) / prev_c["avg_tokens"] * 100
            arrow = "\u2191" if change > 5 else "\u2193" if change < -5 else "\u2192"
            sign = "+" if change >= 0 else ""
            trend_parts.append(f"{sign}{change:.0f}% tokens")
        if prev_c["avg_primitives"] > 0:
            change = (curr_c["avg_primitives"] - prev_c["avg_primitives"]) / prev_c["avg_primitives"] * 100
            sign = "+" if change >= 0 else ""
            trend_parts.append(f"{sign}{change:.0f}% reasoning")
        if trend_parts:
            lines.append(f"  Complexity: {arrow} {', '.join(trend_parts)}")
    elif complexity:
        row = complexity[-1]
        lines.append(f"  Complexity: avg {_format_number(row['avg_tokens'])} tokens, "
                     f"{row['avg_primitives']:.1f} primitives")

    # Accept rate trend (compare last two months)
    standards = standards_evolution(conn)
    if standards:
        curr_s = standards[-1]
        pct = curr_s["accept_rate"] * 100
        line = f"{pct:.0f}%"
        if len(standards) >= 2:
            prev_s = standards[-2]
            delta = (curr_s["accept_rate"] - prev_s["accept_rate"]) * 100
            if abs(delta) >= 1:
                sign = "+" if delta >= 0 else ""
                note = ""
                if delta < -5:
                    note = ", standards rising"
                elif delta > 5:
                    note = ", accepting more"
                line += f" ({sign}{delta:.0f}pp{note})"
        lines.append(f"  Accept rate: {line}")

    # New domains (first seen in the most recent month of data)
    emergence = capability_emergence(conn)
    if emergence and dist:
        latest_month = sorted(dist.keys())[-1]
        new_domains = [r["domain"] for r in emergence if r["first_seen"][:7] == latest_month]
        if new_domains:
            lines.append(f"  New domains: {', '.join(new_domains)}")

    conn.close()

    if lines:
        print("\nTrajectory Highlights")
        print("-" * 30)
        for line in lines:
            print(line)
        print()


def generate_report(months: int | None = None, domain: str | None = None) -> None:
    """Generate and print the full trajectory report."""
    from bespoke.db.init import get_connection

    conn = get_connection()

    # Check if any classified data exists
    has_data = conn.execute(
        "SELECT COUNT(*) as n FROM interactions WHERE domain IS NOT NULL"
    ).fetchone()["n"]

    if has_data == 0:
        print("No classified interactions found.")
        print("Run 'bespoke extract' first to classify your interactions.")
        conn.close()
        return

    # Header
    date_range = conn.execute(
        "SELECT MIN(captured_at) as earliest, MAX(captured_at) as latest "
        "FROM interactions WHERE domain IS NOT NULL"
    ).fetchone()
    earliest = date_range["earliest"][:7]
    latest = date_range["latest"][:7]

    f_clause, f_params = _build_filters(months, domain)
    total = conn.execute(
        f"SELECT COUNT(*) as n FROM interactions WHERE domain IS NOT NULL {f_clause}",
        f_params,
    ).fetchone()["n"]

    print()
    print("BESPOKE \u2014 Trajectory Report")
    print("=" * 30)
    filter_parts = []
    if months:
        filter_parts.append(f"last {months} months")
    if domain:
        filter_parts.append(f"domain: {domain}")
    if filter_parts:
        print(f"Filter: {', '.join(filter_parts)}")
    print(f"Period: {earliest} \u2192 {latest}")
    print(f"Total interactions: {total}")

    # Domain Distribution
    dist = domain_distribution(conn, months)
    if dist and domain is None:  # skip if filtering to one domain
        print()
        print("Domain Distribution")
        # Find max count for bar scaling
        max_count = max(c for by_domain in dist.values() for c in by_domain.values())
        for month in sorted(dist):
            domains_sorted = sorted(dist[month].items(), key=lambda x: -x[1])
            parts = [f"{d} {_bar(c, max_count)} {c}" for d, c in domains_sorted]
            print(f"  {month}  {'   '.join(parts)}")

    # Complexity Trajectory
    complexity = complexity_trajectory(conn, months, domain)
    if complexity:
        print()
        print("Complexity Trajectory")
        for row in complexity:
            tokens_str = _format_number(row["avg_tokens"])
            prims_str = f"{row['avg_primitives']:.1f}"
            print(f"  {row['month']}  avg {tokens_str} tokens   "
                  f"{prims_str} reasoning primitives/interaction")
        if len(complexity) >= 2:
            first, last = complexity[0], complexity[-1]
            trend_parts = []
            if first["avg_tokens"] > 0:
                token_change = (last["avg_tokens"] - first["avg_tokens"]) / first["avg_tokens"] * 100
                sign = "+" if token_change >= 0 else ""
                trend_parts.append(f"{sign}{token_change:.0f}% token complexity")
            if first["avg_primitives"] > 0:
                prim_change = (last["avg_primitives"] - first["avg_primitives"]) / first["avg_primitives"] * 100
                sign = "+" if prim_change >= 0 else ""
                trend_parts.append(f"{sign}{prim_change:.0f}% reasoning density")
            if trend_parts:
                print(f"  Trend: {', '.join(trend_parts)}")

    # Standards Evolution
    standards = standards_evolution(conn, months, domain)
    if standards:
        print()
        print("Standards Evolution")
        for row in standards:
            pct = row["accept_rate"] * 100
            print(f"  {row['month']}  accept rate: {pct:.0f}%  ({row['accepts']}/{row['total']})")
        if len(standards) >= 2:
            first_rate = standards[0]["accept_rate"] * 100
            last_rate = standards[-1]["accept_rate"] * 100
            delta = last_rate - first_rate
            sign = "+" if delta >= 0 else ""
            annotation = ""
            if delta < -5:
                annotation = " (your standards are rising)"
            elif delta > 5:
                annotation = " (accepting more output)"
            print(f"  Trend: {sign}{delta:.0f}% accept rate{annotation}")

    # Capability Emergence
    if domain is None:  # skip if filtering to one domain
        emergence = capability_emergence(conn)
        if emergence:
            print()
            print("Capability Emergence")
            # Group by month of first_seen
            by_month: dict[str, list[str]] = {}
            for row in emergence:
                month = row["first_seen"][:7]
                if month not in by_month:
                    by_month[month] = []
                by_month[month].append(row["domain"])
            for month in sorted(by_month):
                domains_str = ", ".join(f"{d} (first seen)" for d in by_month[month])
                print(f"  {month}  {domains_str}")

    print()
    conn.close()
