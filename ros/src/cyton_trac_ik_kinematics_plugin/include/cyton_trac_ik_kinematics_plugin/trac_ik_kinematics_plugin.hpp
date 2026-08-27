#pragma once

/**
 * @file trac_ik_kinematics_plugin.hpp
 * @brief A fresh, MoveIt2-native kinematics::KinematicsBase implementation
 * wrapping TRAC-IK's core solver (third_party/trac_ik_lib, vendored and
 * modified -- see that directory's own README.md).
 *
 * This is not a mechanical port of trac_ik_kinematics_plugin.cpp (that
 * file, and trac_ik_ros.cpp, are ROS1/catkin code depending on
 * roscpp/tf_conversions/the ROS1 parameter server, none of which exist in
 * ROS2). The core solver itself (trac_ik.cpp/nlopt_ik.cpp/kdl_tl.cpp; see
 * TRAC_IK::TRAC_IK below) has no ROS dependency at all, just
 * KDL/Eigen/Boost/NLopt, and is compiled directly from
 * third_party/trac_ik_lib/src (this package's CMakeLists.txt), the
 * same pattern as cyton_hardware compiling dynamixel_motor.cpp or
 * cyton_ndi_capture compiling ndicapi.
 *
 * The KDL::Chain and joint limits this solver needs are built directly
 * from the MoveIt RobotModel's own parsed URDF (via kdl_parser), the same
 * approach the installed KDLKinematicsPlugin uses. This is what makes it
 * possible to skip trac_ik_ros.cpp's ROS1 parameter-server URDF loading
 * entirely.
 */

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

/**
 * @brief MoveIt kinematics plugin backed by TRAC-IK instead of KDL.
 * See this file's own header comment for why it exists and how it is
 * built. All getPositionIK()/searchPositionIK() overloads below share one
 * implementation, searchPositionIKImpl(); see its own comment for the
 * specifics not covered by kinematics::KinematicsBase's own documented
 * contract (per-call timeout, consistency limits, single-tip-only
 * support).
 */
class TracIKKinematicsPlugin : public kinematics::KinematicsBase
{
public:
  TracIKKinematicsPlugin() = default;

  /**
   * @brief Builds the KDL::Chain and joint limits for `group_name` from
   * `robot_model`'s own parsed URDF, and constructs the TRAC-IK and KDL
   * FK solvers.
   * @param node Unused; required by the KinematicsBase interface.
   * @param robot_model MoveIt RobotModel to build the chain from.
   * @param group_name Planning group name.
   * @param base_frame Chain base frame.
   * @param tip_frames Exactly one tip frame; this plugin does not support
   *        multi-tip groups.
   * @param search_discretization Unused by TRAC-IK; stored via
   *        storeValues() for interface compliance.
   * @return True on success. False if `tip_frames` does not contain
   *         exactly one entry, the RobotModel has no parsed URDF, no
   *         chain could be extracted from base_frame to tip_frames[0], or
   *         `group_name` is not a known JointModelGroup.
   */
  bool initialize(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModel& robot_model,
                   const std::string& group_name, const std::string& base_frame,
                   const std::vector<std::string>& tip_frames, double search_discretization) override;

  /**
   * @brief Single IK solve with the default solver timeout. Delegates to
   * searchPositionIKImpl().
   */
  bool getPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                      std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
                      const kinematics::KinematicsQueryOptions& options =
                          kinematics::KinematicsQueryOptions()) const override;

  /**
   * @brief IK search with a caller-supplied timeout. Delegates to
   * searchPositionIKImpl(); see its comment for why `timeout` is accepted
   * but not actually threaded into the solver.
   */
  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, std::vector<double>& solution,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  /**
   * @brief IK search with consistency limits. Delegates to
   * searchPositionIKImpl(); see its comment for why `consistency_limits`
   * is accepted but not enforced (TRAC-IK has no native concept of it).
   */
  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, const std::vector<double>& consistency_limits,
                        std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  /**
   * @brief IK search with a solution-validation callback. Delegates to
   * searchPositionIKImpl(); `solution_callback`, if provided, can reject
   * an otherwise-valid solution by setting `error_code` to something
   * other than SUCCESS.
   */
  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, std::vector<double>& solution, const IKCallbackFn& solution_callback,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  /**
   * @brief IK search with both consistency limits and a solution-
   * validation callback. Delegates to searchPositionIKImpl(). See the
   * two overloads above for the caveats on each.
   */
  bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                        double timeout, const std::vector<double>& consistency_limits,
                        std::vector<double>& solution, const IKCallbackFn& solution_callback,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options =
                            kinematics::KinematicsQueryOptions()) const override;

  /**
   * @brief Forward kinematics for a list of link names, via this chain's
   * own KDL::ChainFkSolverPos_recursive.
   * @param link_names Names of links in this chain to compute poses for.
   * @param joint_angles Joint angles, in getJointNames() order.
   * @param[out] poses One pose per entry in `link_names`, on success.
   * @return True on success. False if not initialized, `joint_angles` has
   *         the wrong length, or any `link_names` entry is not a segment
   *         in this chain.
   */
  bool getPositionFK(const std::vector<std::string>& link_names, const std::vector<double>& joint_angles,
                      std::vector<geometry_msgs::msg::Pose>& poses) const override;

  const std::vector<std::string>& getJointNames() const override;
  const std::vector<std::string>& getLinkNames() const override;

private:
  /**
   * @brief Shared implementation behind every searchPositionIK() and
   * getPositionIK() overload above. All of them ultimately reduce to
   * "one seed, one target pose, one timeout, optional consistency limits
   * and validation callback", solved via TRAC_IK::TRAC_IK::CartToJnt().
   *
   * @param ik_pose Target pose in base_frame.
   * @param ik_seed_state Seed joint state, in getJointNames() order.
   * @param timeout Unused: TRAC-IK's own per-call timeout is not
   *        threaded through here; the solver instead uses the timeout it
   *        was constructed with (default_timeout_solver_, set once in
   *        initialize()).
   * @param consistency_limits Unused: TRAC-IK has no native
   *        consistency-limit concept, so this is accepted for interface
   *        compatibility but not enforced. May be empty.
   * @param[out] solution The IK solution, in getJointNames() order, on
   *        success.
   * @param solution_callback If set, called with the found solution
   *        before returning success; can reject it by setting
   *        `error_code` to something other than SUCCESS.
   * @param[out] error_code Set to FAILURE if not initialized or the seed
   *        state has the wrong length, NO_IK_SOLUTION if TRAC-IK found no
   *        solution, or SUCCESS/whatever `solution_callback` sets it to
   *        otherwise.
   * @return True only if a solution was found and, if `solution_callback`
   *         was provided, accepted by it.
   */
  bool searchPositionIKImpl(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                            double timeout, const std::vector<double>& consistency_limits,
                            std::vector<double>& solution, const IKCallbackFn& solution_callback,
                            moveit_msgs::msg::MoveItErrorCodes& error_code) const;

  bool initialized_ = false;
  KDL::Chain kdl_chain_;
  KDL::JntArray joint_min_, joint_max_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> link_names_;

  /**
   * @brief The TRAC-IK solver itself. Marked mutable because
   * searchPositionIK()'s own interface is const (per KinematicsBase), but
   * TRAC_IK::CartToJnt() is not itself const; same reasoning as any other
   * MoveIt kinematics plugin wrapping a stateful solver behind a const
   * query API.
   */
  mutable std::unique_ptr<TRAC_IK::TRAC_IK> trac_ik_solver_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;

  /**
   * @brief Solver construction timeout, in seconds. Matches this
   * project's existing kinematics.yaml value for the KDL plugin, for a
   * like-for-like comparison between the two solvers.
   */
  double default_timeout_solver_ = 0.05;
};

}  // namespace cyton_trac_ik_kinematics_plugin
