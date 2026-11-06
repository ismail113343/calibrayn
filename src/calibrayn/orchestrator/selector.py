from calibrayn.scheduler.sm2 import is_due


def choose_chunk(order, schedules, now, recalls=()):
    if not order:
        return None
    known = set(order)
    for chunk_id in recalls:
        if chunk_id in known:
            return chunk_id

    due = [
        chunk_id for chunk_id in order if chunk_id in schedules and is_due(schedules[chunk_id], now)
    ]
    relearning = [chunk_id for chunk_id in due if schedules[chunk_id].is_relearning]
    if relearning:
        return min(relearning, key=lambda chunk_id: schedules[chunk_id].due_at)
    if due:
        return min(due, key=lambda chunk_id: schedules[chunk_id].due_at)

    for chunk_id in order:
        if chunk_id not in schedules:
            return chunk_id

    return min(order, key=lambda chunk_id: schedules[chunk_id].due_at)
