# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
def select_display_window(series, policy="series_percentile"):
    """Fixed per-Series choices; full range preserves highs but reduces contrast.

    Neither the native cache nor existing checkpoint defaults are changed.
    Full range is an input candidate, not a demonstrated classification gain.
    """
    if policy in ("series_percentile", "series_companded"):
        return tuple(series["intensity"]["display_window_uint16"])
    if policy == "native_full_range":
        return (0.0, 65535.0)
    raise ValueError("unknown native intensity window policy: " + str(policy))
