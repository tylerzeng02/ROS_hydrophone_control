#pragma once

// A fresh, MoveIt2-native kinematics::KinematicsBase implementation wrapping
// TRAC-IK's core solver (external/trac_ik/trac_ik_lib), not a mechanical
// port of trac_ik_kinematics_plugin.cpp (that file, and trac_ik_ros.cpp, are
// ROS1/catkin code depending on roscpp/tf_conversions/the ROS1 parameter
// server, none of which exist in ROS2). The core solver itself
// (trac_ik.cpp/nlopt_ik.cpp/kdl_tl.cpp; see TRAC_IK::TRAC_IK below) has no
// ROS dependency at all, just KDL/Eigen/Boost/NLopt, and is compiled
// directly from external/trac_ik/trac_ik_lib/src (this package's
// CMakeLists.txt), the same pattern as cyton_hardware compiling
// dynamixel_motor.cpp or cyton_ndi_capture compiling ndicapi. It is not a
// copy, so it can never drift from the vendored submodule.
//
// The KDL::Chain and joint limits this solver needs are built directly
// from the MoveIt RobotModel's own parsed URDF (via kdl_parser), the same
// approach the installed KDLKinematicsPlugin uses. This is what makes it
// possible to skip trac_ik_ros.cpp's ROS1 parameter-server URDF loading
// entirely.

#include <memory>
#include <string>
#include <vector>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/jntarray.hpp>

#include <moveit/kinematics_base/kinematics_base.hpp>

#include <trac_ik/trac_ik.hpp>

namespace cyton_trac_ik_kinematics_plugin
{

class TracIKKinematicsPlugin : public kinematics::KinematicsBase
{
public:
  TracIKKinematicsPlugin() = default;

  bool initialize(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModel& robot_model,
                   const std::string& group_name, const std::string& base_frame,
                   const std::vector<std::string>& tip_frames, double search_discretization) override;

  bool getPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                      std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
                      const kinematics::KinematicsQueryOptions& options =
                          kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, std::vector<double>& solution,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, const std::vector<double>& consistency_limits,
                        std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, std::vector<double>& solution, const IKCallbackFn& solution_callback,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, const std::vector<double>& consistency_limits,
                        std::vector<double>& solution, const IKCallbackFn& solution_callback,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  bool getPositionFK(const std::vector<std::string>& link_names, const std::vector<double>& joint_angles,
                      std::vector<geometry_msgs::msg::Pose>& poses) const override;

  const std::vector<std::string>& getJointNames() const override;
  const std::vector<std::string>& getLinkNames() const override;

private:
  // Shared implementation behind every searchPositionIK/getPositionIK
  // overload above. All of them ultimately reduce to "one seed, one
  // target pose, one timeout, optional consistency limits and validation
  // callback". consistency_limits may be empty (unused).
  bool searchPositionIKImpl(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                            double timeout, const std::vector<double>& consistency_limits,
                            std::vector<double>& solution, const IKCallbackFn& solution_callback,
                            moveit_msgs::msg::MoveItErrorCodes& error_code) const;

  bool initialized_ = false;
  KDL::Chain kdl_chain_;
  KDL::JntArray joint_min_, joint_max_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> link_names_;

  // mutable: searchPositionIK's own interface is const (per KinematicsBase),
  // but TRAC_IK::CartToJnt is not itself const. Same reasoning as any
  // other MoveIt kinematics plugin wrapping a stateful solver behind a
  // const query API.
  mutable std::unique_ptr<TRAC_IK::TRAC_IK> trac_ik_solver_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;

  double default_timeout_solver_ = 0.05;  // matches this project's existing kinematics.yaml value for KDL
};

}  // namespace cyton_trac_ik_kinematics_plugin
