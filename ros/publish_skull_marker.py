#!/usr/bin/env python3
"""Publishes the skull mesh as a plain RViz visualization Marker -- NOT a
MoveIt collision object. This deliberately bypasses MoveIt's planning scene
entirely, so the mesh is purely visual: MoveIt has no awareness of it (no
collision checking against it, and critically, none of the collision-
geometry processing that's been crashing RViz when the mesh is added via
the Scene Objects tab and then repositioned).

Uses transient-local QoS and keeps republishing periodically, so RViz
displays it correctly whether the Marker display was added before or after
this script starts, and keeps showing it even if RViz gets restarted.

Usage:
    python3 publish_skull_marker.py [--mesh PATH] [--frame FRAME]
        [--xyz X Y Z] [--rpy R P Y] [--scale S] [--alpha A]

Then in RViz: Add -> By display type -> Marker, set its topic to
/skull_marker (the default below).

Defaults to the already-decimated mesh (20.7k triangles) to keep RViz's
render load light -- this script itself doesn't care about mesh size (it
just publishes a resource path, no parsing), but a lighter mesh is still
nicer to actually look at interactively.
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


class SkullMarkerPublisher(Node):
    def __init__(self, args):
        super().__init__("skull_marker_publisher")
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(Marker, "/skull_marker", qos)
        self.args = args
        self.timer = self.create_timer(1.0, self.publish_marker)
        self.get_logger().info(
            f"Publishing mesh '{args.mesh}' as a visual-only Marker on "
            f"/skull_marker (frame '{args.frame}'). Add an RViz Marker "
            f"display subscribed to that topic to see it. Ctrl+C to stop "
            f"(and remove it from RViz)."
        )

    def publish_marker(self):
        m = Marker()
        m.header.frame_id = self.args.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "skull"
        m.id = 0
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = "file://" + self.args.mesh
        m.mesh_use_embedded_materials = False

        m.pose.position.x, m.pose.position.y, m.pose.position.z = self.args.xyz
        qx, qy, qz, qw = euler_to_quaternion(*self.args.rpy)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        m.scale.x = m.scale.y = m.scale.z = self.args.scale

        m.color.r, m.color.g, m.color.b = 0.9, 0.85, 0.8  # bone-ish white
        m.color.a = self.args.alpha

        self.publisher.publish(m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh", default="/home/temp/CT5_DICOM_skull_mesh.stl",
        help="Absolute path to the mesh file. Full-resolution (206,983 tri) "
        "version, not the old _decimated one (20,697 tri) -- same anatomy/"
        "frame (bounds match to <0.1mm), just more detail; kept in sync "
        "with fus_targeting_gui's own default_config.yaml mesh.default_path."
    )
    parser.add_argument(
        "--frame", default="base_link",
        help="TF frame the marker's pose is expressed in."
    )
    parser.add_argument(
        "--xyz", nargs=3, type=float, default=[0.02, 0.27, 0.15], metavar=("X", "Y", "Z"),
        help="Position in meters. Rotated 90deg around Z "
             "(was [-0.27, 0.02, 0.15]) -- same radius/height from "
             "base_link, moved to an adjacent side -- keep in sync with "
             "fus_targeting_gui's default_config.yaml registration.xyz_m."
    )
    parser.add_argument(
        "--rpy", nargs=3, type=float, default=[3.14159, 0.0, 0.0], metavar=("R", "P", "Y"),
        help="Orientation as roll/pitch/yaw in radians."
    )
    parser.add_argument(
        "--scale", type=float, default=0.001,
        help="Uniform scale factor. If the mesh appears ~1000x too big/small, "
             "your STL is likely in mm not meters -- try 0.001 or 1000."
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Opacity, 0 (invisible) to 1 (opaque)."
    )
    args = parser.parse_args()

    rclpy.init()
    node = SkullMarkerPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
