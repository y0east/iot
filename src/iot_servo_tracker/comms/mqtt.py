"""MQTT command and status helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from iot_servo_tracker.common.config import MqttConfig
from iot_servo_tracker.common.packets import CommandPacket, StatusAck


@dataclass(frozen=True)
class MqttTopics:
    command: str
    status: str

    @classmethod
    def from_config(cls, config: MqttConfig) -> "MqttTopics":
        return cls(command=config.command_topic, status=config.status_topic)


@dataclass
class InMemoryMqttBus:
    """Tiny test bus with MQTT-like publish/subscribe semantics."""

    subscribers: dict[str, list[Callable[[str], None]]] = field(default_factory=dict)

    def subscribe(self, topic: str, handler: Callable[[str], None]) -> None:
        self.subscribers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: str) -> None:
        for handler in self.subscribers.get(topic, []):
            handler(payload)


class CommandPublisher:
    def __init__(self, bus: InMemoryMqttBus, topics: MqttTopics) -> None:
        self.bus = bus
        self.topics = topics

    def publish(self, command: CommandPacket) -> None:
        self.bus.publish(self.topics.command, command.to_json())


class StatusPublisher:
    def __init__(self, bus: InMemoryMqttBus, topics: MqttTopics) -> None:
        self.bus = bus
        self.topics = topics

    def publish(self, status: StatusAck) -> None:
        self.bus.publish(self.topics.status, status.to_json())


def build_paho_client(config: MqttConfig):
    """Create a connected paho-mqtt client when the optional dependency exists."""

    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install '.[web,edge]'") from exc

    client = mqtt.Client()
    client.connect(config.host, config.port, keepalive=30)
    return client
