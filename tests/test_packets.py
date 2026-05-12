import unittest

from iot_servo_tracker.common.packets import CommandPacket, CommandType, PacketError


class PacketTests(unittest.TestCase):
    def test_command_round_trip(self) -> None:
        command = CommandPacket.create(CommandType.TRACK, query="red cup")
        decoded = CommandPacket.from_json(command.to_json())
        self.assertEqual(decoded.cmd_type, CommandType.TRACK)
        self.assertEqual(decoded.query, "red cup")

    def test_empty_track_query_is_rejected(self) -> None:
        with self.assertRaises(PacketError):
            CommandPacket.create(CommandType.TRACK, query=" ")


if __name__ == "__main__":
    unittest.main()
