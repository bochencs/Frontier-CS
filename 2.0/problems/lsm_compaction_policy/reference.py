def init(config):
    return {}


def choose(state, observation):
    levels = observation["levels"]
    if levels[0]["runs"] >= 6:
        return {"type": "compact", "level": 0, "pick": "oldest", "max_runs": 4}
    for level in levels[:-1]:
        if level["bytes"] > 1.15 * level["target_bytes"]:
            return {"type": "compact", "level": level["level"], "pick": "largest", "max_runs": 3}
    return {"type": "none"}
