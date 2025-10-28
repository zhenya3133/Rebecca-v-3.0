def promote_event_to_fact(stats) -> bool:
    return (
        stats.freq >= 3
        and stats.trust >= 0.75
        and stats.variance <= 0.15
        and stats.contradictions == 0
    )


def promote_pattern_to_procedure(p) -> bool:
    return p.runs >= 5 and p.delta_quality >= 0.12
