from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConnectorDescriptor:
    connector_type: str
    display_name: str
    capabilities: list[str]
    transport: str = "http"


@dataclass
class ConnectorRegistry:
    descriptors: dict[str, ConnectorDescriptor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.descriptors:
            return
        for descriptor in [
            ConnectorDescriptor("vscode", "Visual Studio Code", ["observe", "act", "feedback", "diagnostics", "files"]),
            ConnectorDescriptor("jetbrains", "JetBrains IDEs", ["observe", "act", "feedback", "diagnostics", "files"]),
            ConnectorDescriptor("neovim", "Neovim", ["observe", "act", "feedback", "files", "commands"]),
            ConnectorDescriptor("zed", "Zed", ["observe", "act", "feedback", "files"]),
            ConnectorDescriptor("codex", "Codex-style clients", ["observe", "act", "feedback", "files", "commands"]),
            ConnectorDescriptor("generic", "Generic IDE Bridge", ["observe", "act", "feedback"]),
        ]:
            self.register(descriptor)

    def register(self, descriptor: ConnectorDescriptor) -> None:
        self.descriptors[descriptor.connector_type] = descriptor

    def list_descriptors(self) -> list[dict[str, object]]:
        return [
            {
                "connector_type": item.connector_type,
                "display_name": item.display_name,
                "capabilities": item.capabilities,
                "transport": item.transport,
            }
            for item in self.descriptors.values()
        ]
