"""Request-owned FLA cache protocol."""


class FLACache:
    """Minimal FLA layer-cache protocol, kept per request, never on the model."""
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index]

    def update(self, layer_idx, offset=1, **state):
        while len(self.states) <= layer_idx:
            self.states.append(None)
        self.states[layer_idx] = state
        return state
