def soft_threshold(threshold, val):
    return [
        (
            each - threshold
            if each > threshold
            else (each + threshold if each < -threshold else 0)
        )
        for each in val
    ]
