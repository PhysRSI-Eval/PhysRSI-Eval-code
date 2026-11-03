"""Site integration TODO, NOT a simulator and NOT a working hardware driver.
Copy to /lizekai/RSIws/real/site_driver.py and implement each method with the site SDK.
See docs/plug_in_charger/REAL_TODO.md. Importing this module performs no physical actions.
"""

class Driver:
    def __init__(self, config):
        self.config = config
        self.metadata = config['metadata']
        raise NotImplementedError('TODO: ARX X5 2023 SDK, camera mapping, calibration, interlocks and native judge')

    def reset(self, layout_id):
        """Acquire operator-prepared scene once; return unique episode_id, instruction, max_episode_steps."""
        raise NotImplementedError

    def observe(self):
        """Return synchronized RGB x3, measured joints x14, link6 xyz x2 and wxyz x2, step counters."""
        raise NotImplementedError

    def act(self, targets, steps):
        """Execute once for 1..5 ticks at 25 Hz; persist hardware ACK before returning.
        Return step_id, executed_steps, terminated, truncated, success (native judge).
        Never retry ambiguous command delivery. Keep actuator safety limits in the SDK controller.
        """
        raise NotImplementedError

    def finish(self, reason):
        """No reset or replay. Return valid, native_success, native_score, native_control_steps, reason."""
        raise NotImplementedError


def make_driver(config):
    return Driver(config)
