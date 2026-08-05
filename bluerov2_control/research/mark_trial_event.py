#!/usr/bin/env python3
"""Publish a task event marker for payload retrieval experiments."""

import argparse
import getpass
import json
import time
from datetime import datetime, timezone
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _wall_time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mark an experiment event that will be recorded by "
            "payload_retrieval_data_logger."
        )
    )
    parser.add_argument(
        "event",
        help=(
            "Event name, for example start, first_detection, approach_start, "
            "hook_attempt, hooked, return_start, docked, success or failure."
        ),
    )
    parser.add_argument("--note", default="", help="Free-text note stored with the event.")
    parser.add_argument("--topic", default="/bluerov2/trial_event", help="std_msgs/String event topic.")
    parser.add_argument("--source", default=getpass.getuser(), help="Event source label.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to publish the event.")
    parser.add_argument(
        "--wait-for-subscriber-sec",
        type=float,
        default=2.0,
        help="Wait this long for the logger before publishing.",
    )
    return parser


class EventPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("mark_trial_event")
        self.publisher = self.create_publisher(String, topic, 10)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    payload = {
        "event": args.event,
        "note": args.note,
        "source": args.source,
        "wall_time_utc": _wall_time(),
    }

    rclpy.init()
    node = EventPublisher(args.topic)
    msg = String()
    msg.data = json.dumps(payload, sort_keys=True)

    try:
        deadline = time.monotonic() + max(0.0, args.wait_for_subscriber_sec)
        while (
            node.publisher.get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.publisher.get_subscription_count() == 0:
            raise RuntimeError(
                "no trial-event subscriber found; start the experiment logger "
                "before marking events"
            )
        for _ in range(max(1, args.repeat)):
            node.publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
